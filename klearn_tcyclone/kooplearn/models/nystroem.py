"""Custom NystroemKernel."""

from kooplearn.models import NystroemKernel

import logging
from typing import Optional, Union

import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, Kernel

from kooplearn._src.operator_regression import dual
from kooplearn._src.utils import NotFittedError, ShapeError, check_is_fitted
from kooplearn.data import TensorContextDataset

logger = logging.getLogger("kooplearn")


class NystroemKernelCustom(NystroemKernel):
    r"""Custom version of Nystroem-accelerated Kernel model minimizing the :math:`L^{2}` loss.

    In this customization, a slide_by argument is included in the fit function, to allow a more flexible handling of time series delays in combination with context lengths larger than two.

    Implements a model approximating the Koopman (deterministic systems) or Transfer (stochastic systems) operator by lifting the state with a *infinite-dimensional nonlinear* feature map associated to a kernel :math:`k` and then minimizing the :math:`L^{2}` loss in the embedded space as described in :footcite:t:`Meanti2023`.

    .. tip::

        The dynamical modes obtained by calling :class:`kooplearn.models.Kernel.modes` correspond to the *Kernel Dynamical Mode Decomposition* by :footcite:t:`Williams2015_KDMD`.

    Args:
        kernel (sklearn.gaussian_process.kernels.Kernel): sklearn Kernel object. Defaults to `DotProduct`.
        reduced_rank (bool): If ``True`` initializes the reduced rank estimator introduced in :footcite:t:`Kostic2022`, if ``False`` initializes the classical principal component estimator.
        rank (int): Rank of the estimator. Defaults to 5.
        tikhonov_reg (float): Tikhonov (ridge) regularization coefficient. ``None`` is equivalent to ``tikhonov_reg = 0``, and internally calls specialized stable algorithms to deal with this specific case.
        svd_solver (str): Solver used to perform the internal SVD calcuations. Currently supported: `full`, uses LAPACK solvers, `arnoldi`, uses ARPACK solvers.
        num_centers (int or float): Number of centers to select. If ``num_centers < 1``, selects ``int(num_centers * n_samples)`` centers. If ``num_centers >= 1``, selects ``int(num_centers)`` centers. Defaults to ``0.1``.
        rng_seed (int): Random Number Generator seed. Only used when ``svd_solver == 'randomized'``.  Defaults to ``None``, that is no explicit seed is setted.

    .. tip::

        A powerful DMD variation proposed by :footcite:t:`Arbabi2017`, known as Hankel-DMD, evaluates the Koopman/Transfer estimators by stacking consecutive snapshots together in a Hankel matrix. When this model is fitted context windows of length > 2, the lookback window length is automatically set to ``context_len - 1``. Upon fitting, the whole lookback window is flattened and *concatenated* together before evaluating the kernel function, thus realizing an Hankel-KDMD estimator.

    Attributes:
        data_fit : Training data, of type ``kooplearn.data.TensorContextDataset``.
        nys_centers_idxs : Indices of the selected centers.
        kernel_X : Kernel matrix evaluated at the subsampled initial states, that is ``self.data_fit[self.nys_centers_idxs, :self.lookback_len, ...]`. Shape ``(n_centers, n_centers)``
        kernel_Y : Kernel matrix evaluated at the subsampled evolved states, that is ``self.data_fit[self.nys_centers_idxs, 1:self.lookback_len + 1, ...]``. Shape ``(n_centers, n_centers)``
        kernel_XY : Cross-kernel matrix between initial and evolved states. Shape ``(n_centers, n_centers)``.
        U : Projection matrix of shape (n_centers, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Meanti2023`).
        V : Projection matrix of shape (n_centers, rank). The Koopman/Transfer operator is approximated as :math:`k(\cdot, X)U V^T k(\cdot, Y)` (see :footcite:t:`Meanti2023`).

    """
    def __init__(
        self,
        kernel: Kernel = DotProduct(),
        reduced_rank: bool = True,
        rank: int = 5,
        tikhonov_reg: Optional[float] = None,
        svd_solver: str = "arnoldi",
        num_centers: Union[float, int] = 0.1,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(
            kernel,
            reduced_rank,
            rank,
            tikhonov_reg,
            svd_solver,
            num_centers,
            rng_seed,
        )
        self._slide_by = -1

    @property
    def slide_by(self) -> int:
        if self._slide_by < 0:
            raise NotFittedError(
                "The slide_by attribute is defined upon fitting the model."
            )
        return self._slide_by

    def fit(self, data: TensorContextDataset, slide_by: int = 1) -> NystroemKernel:
        """Fits the Nystroem Kernel model.

        .. attention::

             The attribute :attr:`lookback_len` will be automatically set to ``data.context_length - 1``. The data will be first flattened along the trailing dimensions before computing the kernel function. The pseudo-code of this operation is

            .. code-block:: python

                X = data.lookback(data.context_length - 1) #Y = data.lookback(data.context_length - slide_by, slide_by=slide_by)
                #Flatten the trailing dimensions
                X = X.reshape(data.shape[0], -1)
                self.kernel_X = self.kernel(X)

        Args:
            data (TensorContextDataset): Dataset of context windows with tensor entries.

        Returns:
            The fitted estimator.
        """
        kernel_Xnys, kernel_Ynys = self._pre_fit_checks(data, slide_by = slide_by)
        if self.reduced_rank:
            U, V, spectral_bias = dual.fit_nystroem_reduced_rank_regression(
                self.kernel_X,
                self.kernel_Y,
                kernel_Xnys,
                kernel_Ynys,
                self.tikhonov_reg,
                self.rank,
                self.svd_solver,
            )
        else:
            U, V, spectral_bias = dual.fit_nystroem_principal_component_regression(
                self.kernel_X,
                self.kernel_Y,
                kernel_Xnys,
                kernel_Ynys,
                self.tikhonov_reg,
                self.rank,
                self.svd_solver,
            )

        self.U = U
        self.V = V
        self.rank = U.shape[1]
        self._spectral_biases = spectral_bias
        # Final Checks
        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "data_fit", "lookback_len"],
        )
        self._is_fitted = True
        logger.info(
            f"Fitted {self.__class__.__name__} model. Lookback length set to {self.lookback_len}"
        )
        return self

    def risk(self, data: Optional[TensorContextDataset] = None) -> float:
        """Risk of the estimator on the validation ``data``.

        Args:
            data (TensorContextDataset or None): Dataset of context windows with tensor entries. If ``None``, evaluates the risk on the training data.

        Returns:
            Risk of the estimator, see Equation 11 of :footcite:p:`Kostic2022` for more details.
        """
        if data is not None:
            if data.context_length != self.data_fit.context_length:
                raise ShapeError(
                    f"The  context length ({data.context_length}) of the validation data does not match the context length of the training data ({self.data_fit.context_length})."
                )
            X_val, Y_val = data.lookback(self.lookback_len), data.lookback(
                self.lookback_len, slide_by=self.slide_by
            )
            X_train, Y_train = (
                self.data_fit.lookback(self.lookback_len)[self.nys_centers_idxs],
                self.data_fit.lookback(self.lookback_len, slide_by=self.slide_by)[
                    self.nys_centers_idxs
                ],
            )
            kernel_Yv = self.kernel(Y_val)
            kernel_XXv = self.kernel(X_train, X_val)
            kernel_YYv = self.kernel(Y_train, Y_val)
        else:
            kernel_Yv = self.kernel_Y
            kernel_XXv = self.kernel_X
            kernel_YYv = self.kernel_Y
        return dual.estimator_risk(
            kernel_Yv, self.kernel_Y, kernel_XXv, kernel_YYv, self.U, self.V
        )

    def eig(
        self,
        eval_left_on: Optional[TensorContextDataset] = None,
        eval_right_on: Optional[TensorContextDataset] = None,
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Returns the eigenvalues of the Koopman/Transfer operator and optionally evaluates left and right eigenfunctions.

        Args:
            eval_left_on (TensorContextDataset or None): Dataset of context windows on which the left eigenfunctions are evaluated.
            eval_right_on (TensorContextDataset or None): Dataset of context windows on which the right eigenfunctions are evaluated.

        Returns:
            Eigenvalues of the Koopman/Transfer operator, shape ``(rank,)``. If ``eval_left_on`` or ``eval_right_on``  are not ``None``, returns the left/right eigenfunctions evaluated at ``eval_left_on``/``eval_right_on``: shape ``(n_samples, rank)``.
        """

        check_is_fitted(
            self,
            ["U", "V", "kernel_X", "kernel_Y", "kernel_YX", "lookback_len", "data_fit"],
        )
        if hasattr(self, "_eig_cache"):
            w, vl, vr = self._eig_cache
        else:
            w, vl, vr = dual.estimator_eig(
                self.U, self.V, self.kernel_X, self.kernel_YX
            )
            self._eig_cache = (w, vl, vr)

        X_fit, Y_fit = self.data_fit.lookback(
            self.lookback_len
        ), self.data_fit.lookback(self.lookback_len, slide_by=self.slide_by)
        X_fit, Y_fit = X_fit[self.nys_centers_idxs], Y_fit[self.nys_centers_idxs]
        if eval_left_on is None and eval_right_on is None:
            # (eigenvalues,)
            return w
        elif eval_left_on is None and eval_right_on is not None:
            # (eigenvalues, right eigenfunctions)
            kernel_Xin_X_or_Y = self.kernel(
                eval_right_on.lookback(self.lookback_len), X_fit
            )
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vr)
        elif eval_left_on is not None and eval_right_on is None:
            # (eigenvalues, left eigenfunctions)
            kernel_Xin_X_or_Y = self.kernel(
                eval_left_on.lookback(self.lookback_len), Y_fit
            )
            return w, dual.evaluate_eigenfunction(kernel_Xin_X_or_Y, vl)
        elif eval_left_on is not None and eval_right_on is not None:
            # (eigenvalues, left eigenfunctions, right eigenfunctions)
            kernel_Xin_X_or_Y_left = self.kernel(
                eval_left_on.lookback(self.lookback_len), Y_fit
            )
            kernel_Xin_X_or_Y_right = self.kernel(
                eval_right_on.lookback(self.lookback_len), X_fit
            )
            return (
                w,
                dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_left, vl),
                dual.evaluate_eigenfunction(kernel_Xin_X_or_Y_right, vr),
            )

    def _pre_fit_checks(self, data: TensorContextDataset, slide_by: int = 1) -> None:
        """Performs pre-fit checks on the training data.

        Initialize the kernel matrices and saves the training data.

        Args:
            data (TensorContextDataset): Dataset of context windows with tensor entries.
        """
        # Save the lookback length
        self._slide_by = slide_by
        self._lookback_len = data.context_length - self.slide_by
        # self._lookback_len = data.context_length - 1
        X_fit, Y_fit = data.lookback(self.lookback_len), data.lookback(
            self.lookback_len, slide_by=self.slide_by
            # self.lookback_len, slide_by=1
        )
        # Perform random center selection
        self.nys_centers_idxs = self.center_selection(X_fit.shape[0])
        X_nys = X_fit[self.nys_centers_idxs]
        Y_nys = Y_fit[self.nys_centers_idxs]

        self.kernel_X, self.kernel_Y, self.kernel_YX = self._init_kernels(X_nys, Y_nys)
        kernel_Xnys, kernel_Ynys = self._init_nys_kernels(X_fit, Y_fit, X_nys, Y_nys)
        self.data_fit = data

        self._check_rank(self.kernel_X.shape[0])
        if hasattr(self, "_eig_cache"):
            del self._eig_cache

        return (
            kernel_Xnys,
            kernel_Ynys,
        )  # Don't need to store them, as they only serve the purpose of fitting.
