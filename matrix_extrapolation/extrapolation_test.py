import numpy as np

from mpmath import expm, extraprec, fsum

from .matrices import allclose_mpmath, Matrix
from .product_formula import ExactMatrixExponential, TrotterFirstOrder
from .extrapolation import RichardsonExtrapolation


def test_RichardsonExtrapolation():
    # testing weights
    assert np.allclose(np.array(RichardsonExtrapolation(ExactMatrixExponential(), [1,2,3]).weights, dtype=float),
                       np.array([1/24, -16/15, 81/40])), ""
    with extraprec(256):
        assert np.isclose(float(fsum(RichardsonExtrapolation(ExactMatrixExponential(), list(range(1, 101))).weights)),
                          1), "Weights must sum to 1."

    t = 0.5
    mats = [Matrix(0.5 * np.identity(2)), Matrix([[0, 1], [1, 0]])]

    # testing with exact exponential
    assert allclose_mpmath(RichardsonExtrapolation(ExactMatrixExponential(), [1,2,3])(t, mats),
                           expm(-1j * (mats[0] + mats[1]) * t)), ""

    # testing with first order Trotter
    assert allclose_mpmath(RichardsonExtrapolation(TrotterFirstOrder(), [1,2,3])(t, mats),
                           (1/24 * expm(-1j * mats[0] * t) @ expm(-1j * mats[1] * t) +
                            -16/15 * (expm(-1j * mats[0] * t / 2) @ expm(-1j * mats[1] * t / 2))**2 +
                            81/40 * (expm(-1j * mats[0] * t / 3) @ expm(-1j * mats[1] * t / 3))**3)), ""
