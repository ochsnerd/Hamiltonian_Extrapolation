from abc import ABC, abstractmethod
from typing import List

from mpmath import eye, mpc, matrix
from mpmath import expm as mp_expm
from numpy import ndarray, identity
from numpy.linalg import matrix_power
from scipy.linalg import expm as scipy_expm

from .matrices import Matrix


class ProductFormula(ABC):
    @abstractmethod
    def __call__(self, time: float, matrices: List[Matrix]) -> Matrix:
        """Return an approximation of e^{-i * sum(matrices) * time}."""
        ...

    @abstractmethod
    def set_steps(self, steps: int) -> None:
        """Set the number of steps used"""
        ...

    def _ensure_dimensions(self, matrices: List[Matrix]) -> None:
        """Make sure that matrices are square and have the same dimensions."""
        if isinstance(matrices[0], matrix):
            rows, cols = matrices[0].rows, matrices[0].cols
            assert all(A.rows == rows and A.cols == cols for A in matrices), (
                "Matrices must be of the same dimension")
        elif isinstance(matrices[0], ndarray):
            rows, cols = matrices[0].shape
            assert all(A.shape == (rows, cols) for A in matrices), (
                "Matrices must be of the same dimension")
        else:
            raise TypeError("Unknown matrix type: {type(matrices[0])}. "
                            "Need mp.matrix or np.ndarray.")

    def _eye(self, matrices: List[Matrix]) -> Matrix:
        """Return the identity matrix matching the matrices.

        In the sense that
        >>> matrices[0] @ self._eye(matrices) == matrices[0]
        True
        (So the type and dimension match)

        Works for square mp.matrix and 2D square np.ndarray.
        """
        if isinstance(matrices[0], matrix):
            rows, cols = matrices[0].rows, matrices[0].cols
            return eye(rows)
        elif isinstance(matrices[0], ndarray):
            rows, cols = matrices[0].shape
            return identity(rows)
        else:
            raise TypeError("Unknown matrix type: {type(matrices[0])}. "
                            "Need mp.matrix or np.ndarray.")

    def _expm(self, s: complex, M: Matrix) -> Matrix:
        """Compute e^(s*M) with the appropriate method

        Either scipy.linalg.expm or mp.expm, depending on
        the type of M.
        """
        if isinstance(M, ndarray):
            return scipy_expm(complex(s) * M)
        elif isinstance(M, matrix):
            # use same method as scipy.linalg.expm
            return mp_expm(s * M, method='pade')
        else:
            raise TypeError("Unknown matrix type: {type(matrices[0])}. "
                            "Need mp.matrix or np.ndarray.")

    def _matpow(self, M: matrix, s: int) -> Matrix:
        """Compute M**s with the type-appropriate method"""
        if isinstance(M, ndarray):
            return matrix_power(M, s)
        elif isinstance(M, matrix):
            # use same method as scipy.linalg.expm
            return M**s
        else:
            raise TypeError("Unknown matrix type: {type(matrices[0])}. "
                            "Need mp.matrix or np.ndarray.")
        

    @abstractmethod
    def __str__(self) -> str:
        ...


class TrotterFirstOrder(ProductFormula):
    r"""Approximate e^{-i \sum H_j t} to first order.

    Computes e^{-i \sum H_j t} as (\prod_{j=1}^J e^{-i H_j t/m})^m.
    """

    def __init__(self, steps: int = 0) -> None:
        """
        Parameters
        ----------
        steps: int
            Number of timesteps taken (m).
        """
        self.m = steps

    def set_steps(self, steps: int) -> None:
        """
        Parameters
        ----------
        steps: int
            Number of timesteps taken (m).
        """
        self.m = steps

    def __call__(self, time: float, matrices: List[Matrix]) -> Matrix:
        r"""Approximate e^{-i * sum(matrices) * time} to first order.

        No optimization for len(matrices) == 2 (symmetric Strang splitting).
        """
        self._ensure_dimensions(matrices)
        result = self._eye(matrices)

        if self.m == 0:
            return result

        t_prime = mpc(time) / self.m

        for H in matrices:
            result = result @ self._expm(-1j * t_prime, H)

        return self._matpow(result, self.m)

    def __str__(self) -> str:
        return "T1"


class TrotterSecondOrder(ProductFormula):
    r"""Approximate e^{-i \sum H_j t} to second order.

    Computes e^{-i \sum H_j t} as
    ((\prod_{j=1}^J e^{-i H_j t/2m})(\prod_{j=J}^1 e^{-i H_j t/2m}))^m.
    """

    def __init__(self, steps: int = 0) -> None:
        """
        Parameters
        ----------
        steps: int
            Number of timesteps taken (m).
        """
        self.m = steps

    def set_steps(self, steps: int) -> None:
        """
        Parameters
        ----------
        steps: int
            Number of timesteps taken (m).
        """
        self.m = steps

    def __call__(self, time: float, matrices: List[Matrix]) -> Matrix:
        r"""Approximate e^{-i * sum(matrices) * time} to second order.

        No optimization for len(matrices) == 2 (symmetric Strang splitting).
        """
        self._ensure_dimensions(matrices)
        result = self._eye(matrices)

        if self.m == 0:
            return result

        if len(matrices) == 1:
            return self._expm(-1j * time * matrices[0])

        t_prime = mpc(time) / (2 * self.m)

        result = result @ self._expm(-2j * t_prime, matrices[0])
        for H in matrices[1:-1]:
            result = result @ self._expm(-1j * t_prime, H)
        result = result @ self._expm(-2j * t_prime, matrices[-1])

        for H in reversed(matrices[1:-1]):
            result = result @ self._expm(-1j * t_prime, H)

        return (self._expm(1j * t_prime, matrices[0]) @
                self._matpow(result, self.m) @
                self._expm(-1j * t_prime, matrices[0]))

    def __str__(self) -> str:
        return "T2"


class ExactMatrixExponential(ProductFormula):
    r"""Compute e^{-i \sum H_j t} exactly"""

    def __init__(self) -> None:
        pass

    def set_steps(self, steps: int = 0) -> None:
        """Exact exponentiation doesn't use steps, arguments are ignored"""
        pass

    def __call__(self, time: float, matrices: List[Matrix]) -> Matrix:
        r"""Compute e^{-i * sum(matrices) * time} exactly by summing up matrices.
        """
        self._ensure_dimensions(matrices)

        return self._expm(-1j * time, sum(matrices))

    def __str__(self) -> str:
        return "EX"
