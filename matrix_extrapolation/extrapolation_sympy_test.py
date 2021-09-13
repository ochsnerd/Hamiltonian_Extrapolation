import sympy as sp
import mpmath as mp

from .product_formula import TrotterSecondOrder
from .extrapolation import RichardsonExtrapolation
from .matrices import Matrix, allclose_mpmath, Pauli

from .extrapolation_sympy import TrotterSecondOrder_sympy, RichardsonExtrapolation_sympy


def test_TrotterSecondOrder_sympy():
    # Sympy doesn't do well with random matrices, use Paulis for testing
    mats_mp = Pauli[1:]
    mats_sp = [sp.Matrix(p) for p in Pauli[1:]]

    for m in range(3):
        s = sp.N(TrotterSecondOrder_sympy(m)(2, mats_sp))
        assert allclose_mpmath(Matrix(s),
                               TrotterSecondOrder(m)(2, mats_mp)), ""


def test_RichardsonExtrapolation_sympy():
    # Sympy doesn't do well with random matrices, use Paulis for testing
    mats_mp = Pauli[1:]
    mats_sp = [sp.Matrix(p) for p in Pauli[1:]]
    m_vector = [1, 2, 3]

    s = sp.N(RichardsonExtrapolation_sympy(0, m_vector)(2, mats_sp))
    assert allclose_mpmath(Matrix(s),
                           RichardsonExtrapolation(TrotterSecondOrder(), m_vector)(2, mats_mp)), ""
