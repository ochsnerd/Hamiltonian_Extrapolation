"""Compute the Richardson extrapolation.

Based in Section V in 'Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation'
"""

from typing import List

from mpmath import fraction, matrix, zeros
from numpy import ndarray, array

from .matrices import Matrix
from .product_formula import ProductFormula, TrotterFirstOrder, TrotterSecondOrder, ExactMatrixExponential


class RichardsonExtrapolation:
    r"""Use a ProductFormula to compute an approximation of e^{-i \sum H_j t}."""

    def __init__(self,
                 product_formula: ProductFormula,
                 pf_steps: List[int]) -> None:
        """
        Parameters
        ----------
        product_formula: ProductFormula
            Used to compute single extrapolation steps.
        pf_steps: List[int]
            Number of steps used for the seperate instances of the product formula.
            len(pf_steps) is 'l' from the paper
        """
        self.pf = product_formula
        self.m_vector = pf_steps
        # Should actually have a baseclass for general symmetric and
        # non-symmetric pfs
        if isinstance(product_formula, TrotterFirstOrder):
            self.weights = self._compute_weights_asymmetric(pf_steps)
        elif isinstance(product_formula, TrotterSecondOrder):
            self.weights = self._compute_weights_symmetric(pf_steps)
        elif isinstance(product_formula, ExactMatrixExponential):
            # use symmetric weights; however exact exponentiation has
            # no error, so different pf_steps don't give different results.
            # everything that matters for the weights is that they sum up
            # to one.
            self.weights = self._compute_weights_symmetric(pf_steps)
        else:
            raise NotImplementedError(f"Unknown product formula {type(product_formula)}")

    def __call__(self, time: float, matrices: List[Matrix]) -> Matrix:
        """Approximate e^{-i * sum(matrices) * time}.

        Parameters
        ----------
        time: float
        matrices: List[Matrix]
            List of sparse matrices (for example scipy.sparse.csc_matrix).
            Used as argument to scipy.sparse.linalg.expm.

        Returns
        -------
        Matrix
        """
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
        assert rows == cols, (
            "Square matrices required")
        # result is mp.matrix either way, since extrapolation is numerically
        # unstable.
        # if arguments are np.ndarray, cast back at the end
        result = zeros(rows)

        for m, a in zip(self.m_vector, self.weights):
            # assuming set_steps to be cheap
            self.pf.set_steps(m)
            result += matrix(self.pf(time, matrices)) * a

        if isinstance(matrices[0], ndarray):
            return array(result.tolist(), dtype=complex)
        return result

    def _compute_weights_symmetric(self, m: List[int]) -> List[float]:
        r"""Compute the extrapolation weights for l extrapolation points.

        For a symmetric product formula with only odd-order terms in the
        BCH expansion. This means we can cancel 2 orders of m (or t) per
        extrapolation step.

        Based on expression (29) in the paper.

        Parameters
        ----------
        m: List[int]
            Number of steps used for the seperate instances of the product formula.
        Returns
        -------
        List[int]
            Extrapolation weights a_j.
        """
        # To compute weights exactly, steps need to be integers
        assert all(isinstance(m_i, int) for m_i in m), "Extrapolation steps have to be integers"
        assert len(m) == len(set(m)), "Extrapolation steps m have to be distinct"

        a = []
        for m_j in m:
            numerator = 1
            denominator = 1
            for m_q in m:
                if m_q == m_j:
                    # will only happen once since m_j are distinct
                    continue
                numerator *= m_j**2
                denominator *= (m_j**2 - m_q**2)

            a += [fraction(numerator, denominator)]

        return a

    def _compute_weights_asymmetric(self, m: List[int]) -> List[float]:
        r"""Compute the extrapolation weights for l extrapolation points.

        For an asymmetric product formula (1st order Trotter for example),
        all terms in the BCH expansion are nonzero. This means we only cancel
        one order in m (or t) per extrapolation step.

        Parameters
        ----------
        m: List[int]
            Number of steps used for the seperate instances of the product formula.
        Returns
        -------
        List[int]
            Extrapolation weights a_j.
        """
        # To compute weights exactly, steps need to be integers
        assert all(isinstance(m_i, int) for m_i in m), "Extrapolation steps have to be integers"
        assert len(m) == len(set(m)), "Extrapolation steps m have to be distinct"

        a = []
        for m_j in m:
            numerator = 1
            denominator = 1
            for m_q in m:
                if m_q == m_j:
                    # will only happen once since m_j are distinct
                    continue
                numerator *= m_j
                denominator *= (m_j - m_q)

            a += [fraction(numerator, denominator)]

        return a
