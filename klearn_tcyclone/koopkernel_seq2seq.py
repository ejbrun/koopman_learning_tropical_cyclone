"""Pytorch implementation of Koopman Kernel Seq2Seq architecture."""

from abc import ABC, abstractmethod

import torch
from kooplearn.data import TensorContextDataset
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define an abstract class
class KoopmanKernelTorch(ABC):
    @abstractmethod
    def __call__(self, X: torch.Tensor, Y: torch.Tensor):
        pass


class RBFKernel(KoopmanKernelTorch):
    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def __call__(self, X: torch.Tensor, Y: torch.Tensor):
        euclidean_squared = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-euclidean_squared / (2 * self.length_scale**2))


class KoopmanKernelSeq2Seq(nn.Module):
    """Koopman Kernel Seq2Seq model.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        kernel: KoopmanKernelTorch,
        input_dim: int,
        input_length: int,
        output_length: int,
        output_dim: int,
        num_steps: int,
        num_nys_centers: int,
        rng_seed: float,
    ):
        """Initialize the Koopman Kernel Seq2Seq model.

        Args:
            kernel (KoopmanKernelTorch): _description_
            input_dim (int): _description_
            input_length (int): _description_
            output_length (int): _description_
            output_dim (int): _description_
            num_steps (int): _description_
            num_nys_centers (int): _description_
            rng_seed (float): _description_
        """
        super().__init__()
        self._kernel = kernel
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_length = output_length
        self.num_steps = num_steps
        self.num_nys_centers = num_nys_centers
        self.rng_seed = rng_seed
        self.rn_generator = torch.Generator()
        _ = self.rn_generator.manual_seed(self.rng_seed)

        self.global_koopman_operator = nn.Parameter(
            torch.empty(
                (self.num_nys_centers, self.num_nys_centers),
                device=device,
                dtype=torch.float32,
            )
        )
        torch.nn.init.xavier_uniform_(
            self.global_koopman_operator
        )  # or any other init method

    # def initialize_nystroem_kernels(self, data_set):
    #     """Initialize the nystroem kernel matrix.

    #     Args:
    #     data_set: dataset to initialize the kernel matrix

    #     Returns:
    #     nystroem_matrix: nystroem kernel matrix
    #     """
    #     rand_indices = self._center_selection(data_set.shape[0], self.rn_generator)

    #     data_set = torch.tensor(data_set, dtype=torch.float32).to(device)
    #     nystroem_matrix = torch.mm(data_set, data_set.t())
    #     return nystroem_matrix

    def _center_selection(
        self, num_pts: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        if self.num_nys_centers < 1:
            num_nys_centers = int(self.num_nys_centers * num_pts)
        else:
            num_nys_centers = int(self.num_nys_centers)

        unif = torch.ones(num_pts)
        rand_indices = unif.multinomial(
            num_nys_centers, replacement=False, generator=generator
        )

        return rand_indices

    def _initialize_nystrom_data(self, tensor_context: TensorContextDataset):
        """First focus on initializing without data_loader object. Maybe need this for later.

        Args:
            tensor_context (_type_): _description_
        """
        n_data = tensor_context.shape[0]
        rand_indices = self._center_selection(n_data, self.rn_generator)

        tensor_context_nys = tensor_context[rand_indices]

        assert tensor_context_nys.context_length == tensor_context.context_length

        tcnys_X = tensor_context_nys.lookback(tensor_context.context_length - 1)
        tcnys_Y = tensor_context_nys.lookback(
            tensor_context.context_length - 1, slide_by=1
        )

        self.nystrom_data_X = torch.tensor(tcnys_X, dtype=torch.float32).to(device)[
            :, tensor_context.context_length // 2
        ]
        # shape: (num_nys_centers, num_feats)
        self.nystrom_data_Y = torch.tensor(tcnys_Y, dtype=torch.float32).to(device)[
            :, tensor_context.context_length // 2
        ]
        # shape: (num_nys_centers, num_feats)
        # We get the middle entry along the context_length axis, as the edge entries
        # might be not so representative since they
        # correspond sometimes to the start and end positions of the TC.

        print(self.nystrom_data_Y.shape, self.nystrom_data_Y.shape)

    # def _initialize_nystrom_data_from_data_loader(self, data_loader):
    #     """First focus on initializing without data_loader object. Maybe need this for later.

    #     Args:
    #         data_loader (_type_): _description_
    #     """
    #     n_data = len(data_loader.dataset)
    #     rand_indices = self._center_selection(n_data, self.rn_generator)

    #     nystrom_data_X = np.array(
    #         [data_loader.dataset[idx][0][0] for idx in rand_indices]
    #     )
    #     nystrom_data_Y = np.array(
    #         [data_loader.dataset[idx][1][0] for idx in rand_indices]
    #     )
    #     self.nystrom_data_X = torch.tensor(nystrom_data_X, dtype=torch.float32).to(
    #         device
    #     )
    #     # shape: (num_nys_centers, num_feats)
    #     self.nystrom_data_Y = torch.tensor(nystrom_data_Y, dtype=torch.float32).to(
    #         device
    #     )
    #     # shape: (num_nys_centers, num_feats)

    def forward(self, inps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            inps (torch.Tensor): input tensor of shape (batch_size, input_length, num_feats)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, input_length, num_feats)
        """
        kernel_nysX_X = self._kernel(self.nystrom_data_X, inps)
        # shape: (batch_size, num_nys_centers, input_length)
        kernel_nysY_Y = self._kernel(self.nystrom_data_Y, self.nystrom_data_Y)
        # shape: (num_nys_centers, num_nys_centers)
        # kernel_nysY_Y = self._kernel(self.nystrom_data_Y, tgts)
        # # shape: (batch_size, num_nys_centers, input_length)

        outs = torch.einsum("ij,bjl->bil", self.global_koopman_operator, kernel_nysX_X)
        # shape: (batch_size, num_nys_centers, input_length)
        outs = torch.einsum("ia,ij,bjl->bla", self.nystrom_data_Y, kernel_nysY_Y, outs)
        # shape: (num_nys_centers, num_feats), (num_nys_centers, num_nys_centers),
        # (batch_size, num_nys_centers, input_length)
        # -> (batch_size, input_length, num_feats)

        return outs


class KoopKernelLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        nystrom_data_Y: torch.Tensor,
        kernel,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.nystrom_data_Y = nystrom_data_Y
        self._kernel = kernel

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of KoopKernelLoss.

        Args:
            input (torch.Tensor): shape (batch_size, input_length, num_feats)
            target (torch.Tensor): shape (batch_size, input_length, num_feats)

        Returns:
            torch.Tensor: _description_
        """
        kernel_nysY_Y = self._kernel(self.nystrom_data_Y, target)
        # shape: (batch_size, num_nys_centers, input_length)

        target_transformed = torch.einsum(
            "ja,bjl->bla", self.nystrom_data_Y, kernel_nysY_Y
        )
        # shape: (num_nys_centers, num_feats), (batch_size, num_nys_centers, input_length)
        # -> (batch_size, input_length, num_feats)

        return F.mse_loss(input, target_transformed, reduction=self.reduction)


def batch_tensor_context(
    tensor_context: TensorContextDataset, batch_size: int, flag_params: dict
):
    tensor_context_inps = torch.tensor(
        tensor_context.lookback(tensor_context.context_length - 1), dtype=torch.float32
    ).to(device)
    tensor_context_tgts = torch.tensor(
        tensor_context.lookback(tensor_context.context_length - 1, slide_by=1),
        dtype=torch.float32,
    ).to(device)

    # tensor_context_torch = torch.tensor(tensor_context.data, dtype=torch.float32).to(
    #     device
    # )

    rand_perm = torch.randperm(tensor_context_inps.shape[0])
    integer_divisor = tensor_context_inps.shape[0] // flag_params["batch_size"]

    tensor_context_inps = tensor_context_inps[rand_perm]
    tensor_context_tgts = tensor_context_tgts[rand_perm]

    tensor_context_inps = tensor_context_inps[
        : integer_divisor * flag_params["batch_size"]
    ]
    tensor_context_tgts = tensor_context_tgts[
        : integer_divisor * flag_params["batch_size"]
    ]

    tensor_context_inps = tensor_context_inps.reshape(
        shape=[
            flag_params["batch_size"],
            integer_divisor,
            tensor_context_inps.shape[1],
            tensor_context_inps.shape[2],
        ]
    )
    tensor_context_tgts = tensor_context_tgts.reshape(
        shape=[
            flag_params["batch_size"],
            integer_divisor,
            tensor_context_tgts.shape[1],
            tensor_context_tgts.shape[2],
        ]
    )

    return tensor_context_inps, tensor_context_tgts
