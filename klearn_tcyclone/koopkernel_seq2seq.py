"""Pytorch implementation of Koopman Kernel Seq2Seq architecture."""

from abc import ABC, abstractmethod

import torch
from kooplearn.data import TensorContextDataset
from torch import nn
from torch.masked import masked_tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define an abstract class
class KoopmanKernelTorch(ABC):
    @abstractmethod
    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        pass


class RBFKernel(KoopmanKernelTorch):
    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        euclidean_squared = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-euclidean_squared / (2 * self.length_scale**2))


class KoopKernelSequencerBase(nn.Module):
    def __init__(
        self,
    ):
        pass


class NystroemKoopKernelSequencer(nn.Module):
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
        context_mode: str = "full_context",
        mask_koopman_operator: bool = False,
        use_nystroem_context_window: bool = False,
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
        self.context_mode = context_mode
        self.use_nystroem_context_window = use_nystroem_context_window
        self.rng_seed = rng_seed
        self.rn_generator = torch.Generator()
        _ = self.rn_generator.manual_seed(self.rng_seed)

        self._mask_koopman_operator(mask_koopman_operator)
        # self.global_koopman_operator = 
        self._initialize_global_koopman_operator(
            self.num_nys_centers,
            self.input_length,
            self.context_mode,
        )

    def _mask_koopman_operator(self, mask_koopman_operator: bool):
        if mask_koopman_operator and self.context_mode != "full_context":
            print("Masking has no effect for context_mode=no_context, last_context.")
        self.mask_koopman_operator = mask_koopman_operator

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
        assert self.input_length + self.output_length == tensor_context.context_length

        tcnys_X = tensor_context_nys.lookback(self.input_length)
        tcnys_Y = tensor_context_nys.lookback(
            self.input_length, slide_by=1
        )
        # tcnys_X = tensor_context_nys.lookback(tensor_context.context_length - 1)
        # tcnys_Y = tensor_context_nys.lookback(
        #     tensor_context.context_length - 1, slide_by=1
        # )

        if self.use_nystroem_context_window:
            self.nystrom_data_X = torch.tensor(tcnys_X, dtype=torch.float32).to(device)
            # shape: (num_nys_centers, input_length, num_feats)
            self.nystrom_data_Y = torch.tensor(tcnys_Y, dtype=torch.float32).to(device)
            # shape: (num_nys_centers, input_length, num_feats)

            nystrom_data_Y_transposed = self.nystrom_data_Y.transpose(0, 1)
            # shape: (input_length, num_nys_centers, num_feats)
            self.kernel_nysY_Y = self._kernel(
                nystrom_data_Y_transposed, nystrom_data_Y_transposed
            ) * self.num_nys_centers ** (-1 / 2)
            # shape: (input_length, num_nys_centers, num_feats), (input_length, num_nys_centers, num_feats)
            # -> (input_length, num_nys_centers, num_nys_centers)
        else:
            self.nystrom_data_X = torch.tensor(tcnys_X, dtype=torch.float32).to(device)[
                :, self.input_length // 2
                # :, tensor_context.context_length // 2
            ]
            # shape: (num_nys_centers, num_feats)
            self.nystrom_data_Y = torch.tensor(tcnys_Y, dtype=torch.float32).to(device)[
                :, self.input_length // 2
                # :, tensor_context.context_length // 2
            ]
            # shape: (num_nys_centers, num_feats)
            # We get the middle entry along the input_length axis, as the edge entries
            # might be not so representative since they
            # correspond sometimes to the start and end positions of the TC.

            self.kernel_nysY_Y = self._kernel(
                self.nystrom_data_Y, self.nystrom_data_Y
            ) * self.num_nys_centers ** (-1 / 2)
            # shape: (num_nys_centers, num_feats), (num_nys_centers, num_feats)
            # -> (num_nys_centers, num_nys_centers)


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

    def _initialize_global_koopman_operator(
        self, num_nys_centers, input_length, context_mode
    ):
        if context_mode == "no_context":
            global_koopman_operator = nn.Parameter(
                torch.empty(
                    (num_nys_centers, num_nys_centers),
                    device=device,
                    dtype=torch.float32,
                )
            )
            koopman_blocks = {}
        elif context_mode == "full_context":
            if self.mask_koopman_operator:
                # global_koopman_operator = None
                # global_koopman_operator = torch.zeros(
                #     (num_nys_centers * input_length, num_nys_centers * input_length),
                #     device=device,
                #     dtype=torch.float32,
                # )

                #=========================================================================

                koopman_blocks = {}
                for i in range(input_length):
                    for j in range(i + 1):
                        koopman_block = nn.Parameter(
                            torch.empty(
                                (num_nys_centers, num_nys_centers),
                                device=device,
                                dtype=torch.float32,
                            )
                        )
                        torch.nn.init.xavier_uniform_(
                            koopman_block
                        )  # or any other init method
                        koopman_blocks[f"{i},{j}"] = koopman_block
                        # global_koopman_operator[
                        #     num_nys_centers * i : num_nys_centers * (i + 1),
                        #     num_nys_centers * j : num_nys_centers * (j + 1),
                        # ] = koopman_block
                global_koopman_operator = None
                # global_koopman_operator = torch.zeros(
                #     (self.num_nys_centers * input_length, self.num_nys_centers * input_length),
                #     device=device,
                #     dtype=torch.float32,
                # )
                # for i in range(input_length):
                #     for j in range(i + 1):
                #         global_koopman_operator[
                #             self.num_nys_centers * i : self.num_nys_centers * (i + 1),
                #             self.num_nys_centers * j : self.num_nys_centers * (j + 1),
                #         ] = koopman_blocks[f"{i},{j}"]

                #=========================================================================

                # global_koopman_operator = torch.zeros(
                #     (num_nys_centers * input_length, num_nys_centers * input_length),
                #     device=device,
                #     dtype=torch.float32,
                # )
                # torch.nn.init.xavier_uniform_(
                #     global_koopman_operator
                # )  # or any other init method

                # mask = torch.ones(global_koopman_operator.shape, dtype=torch.bool, device=device)
                # for i in range(input_length):
                #     for j in range(i+1, input_length):
                #         masked_block = torch.zeros((num_nys_centers, num_nys_centers), dtype=torch.bool)
                #         mask[
                #             num_nys_centers * i : num_nys_centers * (i + 1),
                #             num_nys_centers * j : num_nys_centers * (j + 1),
                #         ] = masked_block

                # global_koopman_operator = masked_tensor(global_koopman_operator, mask, requires_grad=True)
                # global_koopman_operator = nn.Parameter(
                #     global_koopman_operator
                # )
                # koopman_blocks = {}

                #=========================================================================

            else:
                global_koopman_operator = nn.Parameter(
                    torch.empty(
                        (
                            num_nys_centers * input_length,
                            num_nys_centers * input_length,
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                )
                koopman_blocks = {}
        elif context_mode == "last_context":
            global_koopman_operator = nn.Parameter(
                torch.empty(
                    (num_nys_centers, num_nys_centers * input_length),
                    device=device,
                    dtype=torch.float32,
                )
            )
            koopman_blocks = {}
        else:
            raise Exception(f"{context_mode} is not a valid context_mode.")

        if not self.mask_koopman_operator:
        # if global_koopman_operator is not None:
            torch.nn.init.xavier_uniform_(
                global_koopman_operator
            )  # or any other init method
        self.global_koopman_operator = global_koopman_operator
        self.koopman_blocks = nn.ParameterDict(
            koopman_blocks
        )

    def _apply_Y_kernel(self, inps):
        if self.context_mode == "full_context":
            if self.output_length == 1:
                if self.use_nystroem_context_window:
                    outs = torch.einsum("lij,blj->bil", self.kernel_nysY_Y, inps)
                    # shape: (input_length, num_nys_centers, num_nys_centers), (batch_size, input_length, num_nys_centers)
                    # -> (batch_size, num_nys_centers, input_length)
                else:
                    outs = torch.einsum("ij,blj->bil", self.kernel_nysY_Y, inps)
                    # shape: (num_nys_centers, num_nys_centers), (batch_size, input_length, num_nys_centers)
                    # -> (batch_size, num_nys_centers, input_length)
            else:
                if self.use_nystroem_context_window:
                    outs = torch.einsum("lij,bljo->bilo", self.kernel_nysY_Y, inps)
                    # shape: (input_length, num_nys_centers, num_nys_centers), (batch_size, input_length, num_nys_centers, output_length)
                    # -> (batch_size, num_nys_centers, input_length, output_length)
                else:
                    outs = torch.einsum("ij,bljo->bilo", self.kernel_nysY_Y, inps)
                    # shape: (num_nys_centers, num_nys_centers), (batch_size, input_length, num_nys_centers, output_length)
                    # -> (batch_size, num_nys_centers, input_length, output_length)
        return outs

    def _apply_nystroem_data_X(self, inps):
        batch_size = inps.shape[0]
        input_length = inps.shape[1]

        if self.use_nystroem_context_window:
            # shape self.nystrom_data_X: (num_nys_centers, input_length, num_feats)
            # shape inps: (batch_size, input_length, num_feats)

            inps_transposed = inps.transpose(0, 1)
            # shape: (input_length, batch_size, num_feats)
            nystrom_data_X_transposed = self.nystrom_data_X.transpose(0, 1)
            # shape: (input_length, num_nys_centers, num_feats)
            kernel_nysX_X = self._kernel(
                nystrom_data_X_transposed, inps_transposed
            ) * self.num_nys_centers ** (-1 / 2)
            # shape: (input_length, num_nys_centers, num_feats), (input_length, batch_size, num_feats)
            # -> (input_length, num_nys_centers, batch_size)
            kernel_nysX_X = kernel_nysX_X.transpose(0, 2)
            # shape: (batch_size, input_length, num_nys_centers)
        else:
            kernel_nysX_X = self._kernel(
                self.nystrom_data_X, inps
            ) * self.num_nys_centers ** (-1 / 2)
            # shape: (num_nys_centers, num_feats), (batch_size, input_length, num_feats)
            # -> (batch_size, num_nys_centers, input_length)

        if self.context_mode == "full_context":
            kernel_nysX_X = kernel_nysX_X.transpose(1, 2)
            # shape: (batch_size, input_length, num_nys_centers)
            kernel_nysX_X = kernel_nysX_X.reshape(
                shape=(batch_size, self.input_length * self.num_nys_centers)
            )
            # shape: (batch_size, input_length * num_nys_centers)
        return kernel_nysX_X
    
    def _apply_nystroem_data_Y(self, inps):
        if self.use_nystroem_context_window:
            # shape self.nystrom_data_X: (num_nys_centers, input_length, num_feats)
            raise NotImplementedError("Not implemented yet.")
        else:
            if self.context_mode == "full_context":
                if self.output_length == 1:
                    outs = torch.einsum("ja,bjl->bla", self.nystrom_data_Y, inps)
                    # shape: (num_nys_centers, num_feats),
                    # (batch_size, num_nys_centers, input_length)
                    # -> (batch_size, input_length, num_feats)
                else:
                    outs = torch.einsum("ja,bjlo->bloa", self.nystrom_data_Y, inps)
                    # shape: (num_nys_centers, num_feats),
                    # (batch_size, num_nys_centers, input_length, output_length)
                    # -> (batch_size, input_length, output_length, num_feats)
        return outs

    def forward(self, inps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            inps (torch.Tensor): input tensor of shape (batch_size, input_length, num_feats)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, input_length, num_feats)
        """
        batch_size = inps.shape[0]
        input_length = inps.shape[1]
        if self.context_mode != "no_context":
            assert input_length == self.input_length

        # TODO rename to kernel_nys_X_inps
        kernel_nysX_X = self._apply_nystroem_data_X(inps)
        
        # # not needed anymore -> replace by self.kernel_nysY_Y
        # kernel_nysY_Y = self._kernel(
        #     self.nystrom_data_Y, self.nystrom_data_Y
        # ) * self.num_nys_centers ** (-1 / 2)
        # # shape: (num_nys_centers, num_feats), (num_nys_centers, num_feats)
        # # -> (num_nys_centers, num_nys_centers)

        if self.context_mode == "no_context":
            if self.output_length == 1:
                outs = torch.einsum(
                    "ki,ij,bjl->bkl",
                    self.kernel_nysY_Y,
                    self.global_koopman_operator,
                    kernel_nysX_X,
                )
                # shape: (num_nys_centers, num_nys_centers), (num_nys_centers, num_nys_centers),
                # (batch_size, num_nys_centers, input_length)
                # -> (batch_size, num_nys_centers, input_length)
                outs = torch.einsum("ja,bjl->bla", self.nystrom_data_Y, outs)
                # shape: (num_nys_centers, num_feats),
                # (batch_size, num_nys_centers, input_length)
                # -> (batch_size, input_length, num_feats)
            else:
                raise NotImplementedError("Not yet implemented.")
        elif self.context_mode == "full_context":
            # moved to self._apply_nystroem_data_X()
            # kernel_nysX_X = kernel_nysX_X.transpose(1, 2)
            # # shape: (batch_size, input_length, num_nys_centers)
            # kernel_nysX_X = kernel_nysX_X.reshape(
            #     shape=(batch_size, self.input_length * self.num_nys_centers)
            # )
            # # shape: (batch_size, input_length * num_nys_centers)


            if self.mask_koopman_operator:
                # global_koopman_operator = self.global_koopman_operator
                global_koopman_operator = torch.zeros(
                    (self.num_nys_centers * input_length, self.num_nys_centers * input_length),
                    device=device,
                    dtype=torch.float32,
                )
                for i in range(input_length):
                    for j in range(i + 1):
                        global_koopman_operator[
                            self.num_nys_centers * i : self.num_nys_centers * (i + 1),
                            self.num_nys_centers * j : self.num_nys_centers * (j + 1),
                        ] = self.koopman_blocks[f"{i},{j}"]
                
                if self.output_length == 1:
                    outs = torch.einsum(
                        "ij,bj->bi", global_koopman_operator, kernel_nysX_X
                    )
                    # shape: (batch_size, input_length * num_nys_centers)
                    outs = outs.reshape(
                        shape=(batch_size, self.input_length, self.num_nys_centers)
                    )
                else:
                    outs = torch.zeros(
                        size=(
                            batch_size,
                            self.input_length * self.num_nys_centers,
                            self.output_length,
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    out = kernel_nysX_X
                    for idx in range(self.output_length):
                        out = torch.einsum("ij,bj->bi", global_koopman_operator, out)
                        outs[:, :, idx] = out
                    # shape: (batch_size, input_length * num_nys_centers, output_length)
                    outs = outs.reshape(
                        shape=(
                            batch_size,
                            self.input_length,
                            self.num_nys_centers,
                            self.output_length,
                        )
                    )
            else:
                if self.output_length == 1:
                    outs = torch.einsum(
                        "ij,bj->bi", self.global_koopman_operator, kernel_nysX_X
                    )
                    # shape: (batch_size, input_length * num_nys_centers)
                    outs = outs.reshape(
                        shape=(batch_size, self.input_length, self.num_nys_centers)
                    )
                else:
                    outs = torch.zeros(
                        size=(
                            batch_size,
                            self.input_length * self.num_nys_centers,
                            self.output_length,
                        ),
                        device=device,
                        dtype=torch.float32,
                    )
                    out = kernel_nysX_X
                    for idx in range(self.output_length):
                        out = torch.einsum("ij,bj->bi", self.global_koopman_operator, out)
                        outs[:, :, idx] = out
                    # shape: (batch_size, input_length * num_nys_centers, output_length)
                    outs = outs.reshape(
                        shape=(
                            batch_size,
                            self.input_length,
                            self.num_nys_centers,
                            self.output_length,
                        )
                    )

            outs = self._apply_Y_kernel(outs)
            outs = self._apply_nystroem_data_Y(outs)

        elif self.context_mode == "last_context":
            if self.output_length == 1:
                kernel_nysX_X = kernel_nysX_X.reshape(
                    shape=(batch_size, self.num_nys_centers * self.input_length)
                )
                outs = torch.einsum(
                    "ij,bj->bi", self.global_koopman_operator, kernel_nysX_X
                )
                # shape: (batch_size, num_nys_centers)
                outs = torch.einsum("ij,bj->bi", self.kernel_nysY_Y, outs)
                # -> (batch_size, num_nys_centers)
                outs = torch.einsum("ja,bj->ba", self.nystrom_data_Y, outs)
                # shape: (num_nys_centers, num_feats),
                # (batch_size, num_nys_centers)
                # -> (batch_size, num_feats)
            else:
                outs = torch.zeros(
                    size=(batch_size, self.num_nys_centers, self.output_length),
                    device=device,
                    dtype=torch.float32,
                )
                out = kernel_nysX_X
                # shape: (batch_size, num_nys_centers, input_length)
                for idx in range(self.output_length):
                    out_flat = out.reshape(
                        shape=(batch_size, self.num_nys_centers * self.input_length)
                    )
                    next_out = torch.einsum(
                        "ij,bj->bi", self.global_koopman_operator, out_flat
                    )
                    # shape: (batch_size, num_nys_centers)
                    out = torch.cat([out[:, :, 1:], next_out.unsqueeze(dim=2)], dim=2)
                    # shape: (batch_size, num_nys_centers, input_length)
                    outs[:, :, idx] = out[:, :, -1]
                # shape: (batch_size, num_nys_centers, output_length)
                outs = torch.einsum("ij,bjo->bio", self.kernel_nysY_Y, outs)
                # -> (batch_size, num_nys_centers, output_length)
                outs = torch.einsum("ja,bjo->boa", self.nystrom_data_Y, outs)
                # shape: (num_nys_centers, num_feats),
                # (batch_size, num_nys_centers, output_length)
                # -> (batch_size, output_length, num_feats)

        return outs


class KoopKernelLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        nystrom_data_Y: torch.Tensor,
        kernel: KoopmanKernelTorch,
        context_mode: str = "no_context",
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.nystrom_data_Y = nystrom_data_Y
        self._kernel = kernel
        self.context_mode = context_mode

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of KoopKernelLoss.

        Args:
            input (torch.Tensor): shape (batch_size, input_length, num_feats)
            target (torch.Tensor): shape (batch_size, input_length, num_feats)

        Returns:
            torch.Tensor: _description_
        """
        # kernel_nysY_Y = self._kernel(self.nystrom_data_Y, target)
        # # shape: (batch_size, num_nys_centers, input_length)

        # target_transformed = torch.einsum(
        #     "ja,bjl->bla", self.nystrom_data_Y, kernel_nysY_Y
        # )
        # # shape: (num_nys_centers, num_feats), (batch_size, num_nys_centers, input_length)
        # # -> (batch_size, input_length, num_feats)

        target_transformed = target

        return F.mse_loss(input, target_transformed, reduction=self.reduction)
