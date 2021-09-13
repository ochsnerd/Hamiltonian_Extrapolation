"""
Error bounds as described in

Andrew M. Childs, Yuan Su, Minh C. Tran, Nathan Wiebe, and Shuchen Zhu.
Theory of Trotter Error with Commutator Scaling.
Physical Review X, 11(1):011020, February 2021.
Publisher: American Physical Society.
"""
import mpmath as mp

from matrix_extrapolation import norm_mpmath, commutator


def zeros_like(mat):
    return mp.zeros(mat.rows, mat.cols)


def trotter1_error_bound(matrices, time):
    """Equation (120) (Tight error bound for the first-order Lie-Trotter formula)"""
    sum_comm_norms = 0
    for i, H_g1 in enumerate(matrices):
        H_sum = zeros_like(matrices[0])
        for H_g2 in matrices[i+1:]:
            H_sum += H_g2
        C = commutator(H_sum, H_g1)
        sum_comm_norms += norm_mpmath(C, ord=2)

    return abs(time)**2 / 2 * sum_comm_norms


def trotter2_error_bound(matrices, time):
    """Equation (121) (Tight error bound for the second-order Suzuki formula)"""
    first_term, second_term = 0, 0
    for i, H_g1 in enumerate(matrices):
        sum_g2 = zeros_like(matrices[0])
        for H_g2 in matrices[i+1:]:
            sum_g2 += H_g2
        C_g2 = commutator(sum_g2, H_g1)

        # the sums over gamma_2 and gamma_3 have the same limits
        # and therefore compute the same thing
        # (is this actually correct?)
        sum_g3 = sum_g2
        C = commutator(sum_g3, C_g2)
        first_term += norm_mpmath(C, ord=2)

        # also the gamma_2 sum in the second term is the same expression,
        # so reuse it here as well (also [A,B] = -[B,A])
        C_g2 = -C_g2
        C = commutator(H_g1, C_g2)
        second_term += norm_mpmath(C, ord=2)

    first_term *= abs(time)**3 / 12
    second_term *= abs(time)**3 / 24
    return first_term + second_term
