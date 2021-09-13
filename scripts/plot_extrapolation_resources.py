import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

from seaborn import color_palette

from qiskit.quantum_info import Pauli

from matrix_extrapolation import (
    BCH_term, compute_extrapolation_error_term,
    S2,
    SympyOperator, PauliOperator,
    norm_mpmath,
    RichardsonExtrapolation, ExactMatrixExponential, TrotterSecondOrder, TrotterFirstOrder)

from util import savefig
from ising_lattice import IsingLatticeHamiltonian


def extrapolation_error_empirical_S1(matrices, time, ms):
    return norm_mpmath(ExactMatrixExponential()(time, matrices) -
                       RichardsonExtrapolation(TrotterFirstOrder(), ms)(time, matrices),
                       use_mpmath=False,
                       ord=2)


def extrapolation_error_empirical_S2(matrices, time, ms):
    return norm_mpmath(ExactMatrixExponential()(time, matrices) -
                       RichardsonExtrapolation(TrotterSecondOrder(), ms)(time, matrices),
                       use_mpmath=False,
                       ord=2)


def max_cnots(pf, ms):
    assert len(lattice_shape) == 1
    if isinstance(pf, str) and pf[:2] == "S1":
        return max(cnots_Ising_S1(m, lattice_shape[0]) for m in ms)
    if isinstance(pf, S2):
        return max(cnots_Ising_S2(m, lattice_shape[0]) for m in ms)
    raise NotImplementedError(f"No resource estimation for {pf}")


def cnots_Ising_S2(m, N):
    # non-periodic chain -> N-1 ZZ terms
    # each ZZ-term of the form CNOT R_Z CNOT
    # -> 2*(N-1) CNOTS for each e^{it H_Z}

    # symmetric Strang splitting,
    # Carrera2020 eq (21)

    # if m==1, H_Z in the middle
    return 2*m*(N-1)


def cnots_Ising_S1(m, N):
    # non-periodic chain -> N-1 ZZ terms
    # each ZZ-term of the form CNOT R_Z CNOT
    # -> 2*(N-1) CNOTS for each e^{it H_Z}
    return 2*m*(N-1)


lattice_shape = (6, )
H = IsingLatticeHamiltonian(lattice_shape)
H_Z = PauliOperator(H.interaction_term_paulis())
H_X = PauliOperator(H.transverse_field_term_paulis())
s2_Ising = S2([H_Z, H_X], "tf_Ising_" + str(lattice_shape).replace(" ", ""))
nonzero_terms = 3
ls = range(1,7)
m1s = [range(1, l+1) for l in ls]
m3s = [[3*m for m in range(1, l+1)] for l in ls]
resource_func = max_cnots

ts = [0.1, 0.5, 1.0]
colors = color_palette("crest", len(ts))

fig = plt.figure()
for t, color in zip(reversed(ts), colors):  # reverse ts so that the legend matches up with the curves
    # no extrapolation
    queriest_S1, queriest_S2 = [], []
    error_empiricalt_S1, error_empiricalt_S2 = [], []
    # error_seriest = []
    for m in range(1, 8):
        queriest_S1 += [resource_func("S1_J=2", [m])]
        error_empiricalt_S1 += [extrapolation_error_empirical_S1((H_Z.to_matrix(), H_X.to_matrix()),
                                                                 t, [m])]

        queriest_S2 += [resource_func(s2_Ising, [m])]
        error_empiricalt_S2 += [extrapolation_error_empirical_S2((H_Z.to_matrix(), H_X.to_matrix()),
                                                                 t, [m])]

    # yes extrapolation
    queries1_S1, queries1_S2 = [], []
    queries3_S1, queries3_S2 = [], []
    error_empirical1_S1, error_empirical1_S2 = [], []
    error_empirical3_S1, error_empirical3_S2 = [], []
    for i, (m1, m3, l) in enumerate(zip(m1s, m3s, ls)):
        queries1_S1 += [resource_func("S1_J=2", m1)]
        queries3_S1 += [resource_func("S1_J=2", m3)]
        error_empirical1_S1 += [extrapolation_error_empirical_S1((H_Z.to_matrix(), H_X.to_matrix()),
                                                                 t, m1)]
        error_empirical3_S1 += [extrapolation_error_empirical_S1((H_Z.to_matrix(), H_X.to_matrix()),
                                                                 t, m3)]

        queries1_S2 += [resource_func(s2_Ising, m1)]
        queries3_S2 += [resource_func(s2_Ising, m3)]
        error_empirical1_S2 += [extrapolation_error_empirical_S2((H_Z.to_matrix(), H_X.to_matrix()),
                                                                 t, m1)]
        error_empirical3_S2 += [extrapolation_error_empirical_S2((H_Z.to_matrix(), H_X.to_matrix()),
                                                                 t, m3)]

    if t == ts[0]:
        plt.plot(queriest_S1[:-1], error_empiricalt_S1[:-1], color='k', marker="P", label=f"${t=}$, $S_1$")
        plt.plot(queries1_S1, error_empirical1_S1, color=color, linestyle='solid', marker="P", label=f"${t=}$" + ", $M^l_{S_1}$")
#         plt.plot(queries3_S1[:-2], error_empirical3_S1[:-2], color=color, linestyle='dotted', marker="P", label=f"${t=}$" + ", $S_1$, $m = \{3, 6, \dots, 3l\}$")
        plt.plot(queriest_S2[:-1], error_empiricalt_S2[:-1], color='k', marker="^", label=f"${t=}$, $S_2$")
        plt.plot(queries1_S2[:-1], error_empirical1_S2[:-1], color=color, linestyle='solid', marker="^", label=f"${t=}$" + ", $M^l_{S_2}$")
#         plt.plot(queries3_S2[:-2], error_empirical3_S2[:-2], color=color, linestyle='dotted', marker="^", label=f"${t=}$" + ", $S_2$, $m = \{3, 6, \dots, 3l\}$")
    else:
        plt.plot(queriest_S1[:-1], error_empiricalt_S1[:-1], color='k', marker="P") # , label=f"${t=}$, $S_2$, empirical")
        plt.plot(queries1_S1, error_empirical1_S1, color=color, linestyle='solid', marker="P", label=f"${t=}$")
#         plt.plot(queries3_S1[:-2], error_empirical3_S1[:-2], color=color, linestyle='dotted', marker="P")
        plt.plot(queriest_S2[:-1], error_empiricalt_S2[:-1], color='k', marker="^") # , label=f"${t=}$, $S_2$, empirical")
        plt.plot(queries1_S2, error_empirical1_S2, color=color, linestyle='solid', marker="^", label=f"${t=}$")
#         plt.plot(queries3_S2[:-2], error_empirical3_S2[:-2], color=color, linestyle='dotted', marker="^")

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xlabel(r'Circuit size (CNOT)')
plt.ylabel(r'$\epsilon$')
plt.xticks([queries1_S1[0]] + list(range(queries1_S1[0] // 5 * 5, queries1_S1[-1] // 5 * 5 + 1, 5)))
# plt.xticks(ls)
# plt.xscale('log')
plt.yscale('log')
savefig(f'extrapolation_resources')
plt.close(fig)

