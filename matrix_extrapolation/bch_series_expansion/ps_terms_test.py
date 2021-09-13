import sympy as sp

from .ps_terms import restricted_k_compositions, ps_prefactor, ps_prefactor_symmetric, compute_ps_terms
from .extrapolation_weights import ExtrapolationWeightsSymmetric


def test_restricted_k_compositions():
    assert tuple(restricted_k_compositions(0, 1, [1])) == ()
    assert tuple(restricted_k_compositions(1, 0, [1])) == ()
    assert tuple(restricted_k_compositions(1, 1, [])) == ()

    assert tuple(restricted_k_compositions(1, 1, [1])) == ((1,),)
    assert tuple(restricted_k_compositions(6, 1, [1, 3, 5])) == ()
    assert tuple(restricted_k_compositions(6, 2, [1, 3, 5])) == (
        (1, 5), (3, 3), (5, 1))
    assert tuple(restricted_k_compositions(6, 3, [1, 3, 5])) == ()
    assert tuple(restricted_k_compositions(6, 4, [1, 3, 5])) == (
        (1, 1, 1, 3), (1, 1, 3, 1), (1, 3, 1, 1), (3, 1, 1, 1))
    assert tuple(restricted_k_compositions(6, 5, [1, 3, 5])) == ()
    assert tuple(restricted_k_compositions(6, 6, [1, 3, 5])) == (
        (1, 1, 1, 1, 1, 1),)


def test_ps_prefactor():
    assert ps_prefactor(3, 1, [], []) == 0
    assert ps_prefactor(11, 1, [1], [1]) == sp.Rational(1, sp.factorial(10))
    assert ps_prefactor(3, 1, [1, 2], [-1, 2]) == 0


def test_ps_prefactor_symmetric():
    assert ps_prefactor_symmetric(3, 1, [], []) == 0
    assert ps_prefactor_symmetric(11, 1, [1], [1]) == sp.Rational(1, sp.factorial(9))
    assert ps_prefactor_symmetric(3, 1, [1, 2], [sp.Rational(-1, 3), sp.Rational(4, 3)]) == 0


def test_compute_ps_terms():
    assert tuple(compute_ps_terms(5, [1], [1], [1, 3, 5])) == (
        (sp.Rational(1, 6), (1, 1, 3)),
        (sp.Rational(1, 6), (1, 3, 1)),
        (sp.Rational(1, 6), (3, 1, 1)),
        (sp.Rational(1, 1), (5,))), "Error expansion of a single second order Trotter step"

    l = 4
    m = range(1, l+1)
    a = ExtrapolationWeightsSymmetric(m)
    assert all(not list(compute_ps_terms(o, m, a, [1, 3, 5, 7])) for o in range(2*l)), (
        "l extrapolation steps cancel the first 2*l terms in the error series")
