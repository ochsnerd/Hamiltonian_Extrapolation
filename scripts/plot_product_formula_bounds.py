import mpmath as mp
import matplotlib.pyplot as plt
from seaborn import color_palette

from matrix_extrapolation import (TrotterSecondOrder,
                                  ExactMatrixExponential,
                                  norm_mpmath,
                                  commutator)

from matrix_extrapolation import (S2, compute_extrapolation_error_term, PauliOperator)


from util import savefig, IsingLatticeHamiltonian

import trotter_bounds_childs2021 as childs2021


def trotter2_error_empirical(matrices, time, m):
    return norm_mpmath(ExactMatrixExponential()(time, matrices) - TrotterSecondOrder(m)(time, matrices),
                       use_mpmath=True,
                       p=2)


def trotter2_error_bound(matrices, time):
    """
    Error bounds as described in

    Andrew M. Childs, Yuan Su, Minh C. Tran, Nathan Wiebe, and Shuchen Zhu.
    Theory of Trotter Error with Commutator Scaling.
    Physical Review X, 11(1):011020, February 2021.
    Publisher: American Physical Society.
    """
    """Equation (121) (Tight error bound for the second-order Suzuki formula)"""
    def zeros_like(mat):
        return mp.zeros(mat.rows, mat.cols)
    
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


def main():
    Ns = [3, 4, 5, 6]
    t = 0.25

    empirical = []
    bound = []

    # order needs to be sequential
    series_order = [3, 4, 5, 6]
    series = [[] for _ in series_order]

    for N in Ns:
        # create Hamiltonian
        lattice_shape = (N, )
        H = IsingLatticeHamiltonian(lattice_shape)
        H_Zm = H.interaction_term_matrix()
        H_Xm = H.transverse_field_term_matrix()

        H_Zp = PauliOperator(H.interaction_term_paulis())
        H_Xp = PauliOperator(H.transverse_field_term_paulis())

        # compute emprirical error
        empirical += [trotter2_error_empirical([H_Zm, H_Xm], t, 1)]

        # compute Childs Bound
        bound += [childs2021.trotter2_error_bound([mp.matrix(H_Zm), mp.matrix(H_Xm)], t)]

        # compute series approximation
        S2_Ising = S2([H_Zp, H_Xp], "tf_Ising_" + str(lattice_shape).replace(" ", ""))
        series_error_mat = sum((-1j * t)**o * compute_extrapolation_error_term(o, S2_Ising, [1]) for o in range(series_order[0]))
        for i, O in enumerate(series_order):
            series_error_mat += (-1j * t)**O * compute_extrapolation_error_term(O, S2_Ising, [1])
            series[i] += [norm_mpmath(series_error_mat.to_matrix(), use_mpmath=True, p=2)]

    for errors, order, c in zip(series, series_order, reversed(color_palette("crest", len(series_order)))):
        plt.plot(Ns, errors, '--', label='Series, ' + str(order - 2) + " nonzero terms", color=c, marker='x')

    plt.plot(Ns, bound, 'k--', label='Bound by Childs et al.')
    plt.plot(Ns, empirical, label='Trotter-Suzuki $S_2$', color='r', marker='o')

    plt.xlabel("$N$")
    plt.ylabel(r"$\epsilon$")
    plt.xticks(Ns)
    plt.legend()

    savefig("product_formula_bounds")


if __name__ == '__main__':
    main()
