import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

from seaborn import color_palette
from math import factorial, prod

from qiskit.quantum_info import Pauli

from matrix_extrapolation import (
    BCH_term, compute_extrapolation_error_term,
    S2,
    PauliOperator,
    norm_mpmath,
    RichardsonExtrapolation, ExactMatrixExponential, TrotterSecondOrder)

from util import savefig, IsingLatticeHamiltonian


def extrapolation_error_empirical(matrices, time, ms):
    return norm_mpmath(ExactMatrixExponential()(time, matrices) -
                       RichardsonExtrapolation(TrotterSecondOrder(), ms)(time, matrices),
                       use_mpmath=False,
                       ord=2)


def conjectured(l, t):
    denom = prod(range(1, l+1))
    denom = factorial(2*l+1)
    return (2*t)**(2*l + 1) / denom


lattice_shape = (3, )
H = IsingLatticeHamiltonian(lattice_shape)
H_Z = PauliOperator(H.interaction_term_paulis())
H_X = PauliOperator(H.transverse_field_term_paulis())
s2_Ising = S2([H_Z, H_X], "tf_Ising_" + str(lattice_shape).replace(" ", ""))
nonzero_terms = 3
ls = range(1,6)

E_l = []
for l in ls:
    m = range(1, l+1)
    E_l += [[compute_extrapolation_error_term(o, s2_Ising, m) for o in range(2*l + 1 + nonzero_terms + 1)]]


ts = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
colors = color_palette("crest", len(ts))

fig = plt.figure()
for t, color in zip(reversed(ts), colors):  # reverse ts so that the legend matches up with the curves
    error_empirical = []
    error_series = []
    for i, l in enumerate(ls):
        m = range(1, l+1)
        error_empirical += [extrapolation_error_empirical((H_Z.to_matrix(), H_X.to_matrix()),
                                                          t, m)]
        E = E_l[i]
        error_series += [norm_mpmath(mp.matrix(
            sum((-1j * t)**i * E_i
                for i, E_i in enumerate(E)).to_matrix()),
                                     use_mpmath=False, ord=2)]

    if t == ts[0]:
        plt.plot(ls, error_series, marker="v", color=color,
                 label=f"${t=},$" + " series")
                 # label=(r'$\norm{\sum_{n = 2l + 1}^{2l + 1 +' + str(nonzero_terms) + r'} (-it) E_n}$, ' + f'${t=}$'))
        plt.plot(ls, error_empirical, marker="^", color=color,
                 label=f"${t=},$" + " empirical")
                 # label=r'$\norm{M_{S_2}^l(t, m = [l]) - e^{-it (H_Z + H_X)}}$, ' + f'${t=}$')
        plt.plot(ls, [error_empirical[0] / t**(2*ls[0]+1) * t**(2*l + 1) for l in ls], 'k:', label=r'$\mathcal{O}(t^{2l + 1})$')
        plt.plot(ls, [error_empirical[0] / conjectured(ls[0], t) * conjectured(l, t) for l in ls], 'k--', label=r'$\mathcal{O}(\frac{(2t)^{2l + 1}}{(2l + 1)!})$')
    else:
        plt.plot(ls, error_series, color=color,  marker="v", label=f"${t=}$")
        plt.plot(ls, error_empirical, color=color, marker="^")
        plt.plot(ls, [error_empirical[0] / t**(2*ls[0]+1) * t**(2*l + 1) for l in ls], 'k:')
        plt.plot(ls, [error_empirical[0] / conjectured(ls[0], t) * conjectured(l, t) for l in ls], 'k--')

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xlabel(r'$l$')
plt.xticks(ls)
plt.ylabel(r'$\epsilon$')
plt.yscale('log')
savefig('series_convergence')
plt.close(fig)
