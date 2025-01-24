"""Pytorch implementation of Koopman Kernel Seq2Seq architecture."""

from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn

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

    def initialize_nystroem_kernels(self, data_set):
        """Initialize the nystroem kernel matrix.

        Args:
        data_set: dataset to initialize the kernel matrix

        Returns:
        nystroem_matrix: nystroem kernel matrix
        """
        rand_indices = self._center_selection(data_set.shape[0], self.rn_generator)

        data_set = torch.tensor(data_set, dtype=torch.float32).to(device)
        nystroem_matrix = torch.mm(data_set, data_set.t())
        return nystroem_matrix

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

    def _initialize_nystrom_data(self, data_loader):
        n_data = len(data_loader.dataset)
        rand_indices = self._center_selection(n_data, self.rn_generator)

        nystrom_data_X = np.array(
            [data_loader.dataset[idx][0][0] for idx in rand_indices]
        )
        nystrom_data_Y = np.array(
            [data_loader.dataset[idx][1][0] for idx in rand_indices]
        )
        self.nystrom_data_X = torch.tensor(nystrom_data_X, dtype=torch.float32).to(
            device
        )
        # shape: (num_nys_centers, num_feats)
        self.nystrom_data_Y = torch.tensor(nystrom_data_Y, dtype=torch.float32).to(
            device
        )
        # shape: (num_nys_centers, num_feats)

    def forward(self, inps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            inps (torch.Tensor): input tensor of shape (batch_size, input_length, num_feats)

        Returns:
            torch.Tensor: output tensor of shape (batch_size, input_length, num_feats)
        """        
        kernel_nysX_X = self._kernel(self.nystrom_data_X, inps)
        # shape: (batch_size, num_nys_centers, input_length)
        outs = torch.einsum("ij,bjl->bil", self.global_koopman_operator, kernel_nysX_X)
        # shape: (batch_size, num_nys_centers, input_length)

        return outs
