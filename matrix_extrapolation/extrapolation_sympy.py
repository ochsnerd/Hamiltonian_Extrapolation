"""
Implementation of Richardson Extrapolation using sympy.
Useful for verification and testing. Not useful for matrices
larger than 2x2 since sympy becomes very slow.
(understandably since it was not designed for numerical computations)
"""
import sympy as sp


def sympy_spectral_norm(A):
    return sp.re(A.singular_values()[0])


def trotter_second_order(mats, t, m):
    """Vazquez2020 Eq (20)"""
    result = sp.eye(mats[0].shape[0])

    if m == 0:
        return result

    t_prime = sp.Rational(t / m) / 2

    for H in mats[:-1]:
        result = result * sp.exp(-sp.I * H * t_prime)

    result = result * sp.exp(-sp.I * mats[-1] * 2 * t_prime)

    for H in reversed(mats[:-1]):
        result = result * sp.exp(-sp.I * H * t_prime)

    return result**m


def richardson_second_order_trotter(m_vector, mats, t):
    # compute weights
    weights = []
    for m_j in m_vector:
        a_j = sp.Rational(1, 1)
        for m_q in m_vector:
            if m_q == m_j:
                continue
            a_j *= sp.Rational(m_j**2, m_j**2 - m_q**2)
        weights += [a_j]

    assert sum(weights) == 1, ""

    result = sp.zeros(mats[0].shape[0])
    for a, m in zip(weights, m_vector):
        result = result + a * trotter_second_order(mats, t, m)

    return result


from typing import List
from .product_formula import TrotterSecondOrder


class TrotterSecondOrder_sympy(TrotterSecondOrder):
    def __call__(self, time: float, matrices: List[sp.Matrix]) -> sp.Matrix:
        return trotter_second_order(matrices, time, self.m)


class RichardsonExtrapolation_sympy:
    """Just mirror the interface of .extrapolation.RichardsonExtrapolation"""
    def __init__(self, product_formula, pf_steps):
        # ignore product_formula argument and hardcode TrotterSecondOrder_sympy
        self.pf_steps = pf_steps

    def __call__(self, time, matrices):
        return richardson_second_order_trotter(self.pf_steps, matrices, time)
