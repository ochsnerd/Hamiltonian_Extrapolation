import sympy as sp

from .extrapolation_weights import ExtrapolationWeights, ExtrapolationWeightsSymmetric


def test_ExtrapolationWeights():
    assert ExtrapolationWeights([1])[0] == 1
    assert ExtrapolationWeights([1, 2])[0] == -1

    assert sum(ExtrapolationWeights(range(1, 101))) == 1


def test_ExtrapolationWeightsSymmetric():
    assert ExtrapolationWeightsSymmetric([1])[0] == 1
    assert ExtrapolationWeightsSymmetric([1, 2])[0] == sp.Rational(-1, 3)

    assert sum(ExtrapolationWeightsSymmetric(range(1, 101))) == 1
