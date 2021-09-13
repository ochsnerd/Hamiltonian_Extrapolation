import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from seaborn import color_palette

from util import savefig

from matrix_extrapolation import (TrotterFirstOrder,
                                  TrotterSecondOrder,
                                  ExactMatrixExponential,
                                  norm_mpmath,
                                  Random_1s_noncommuting_MatrixGenerator)


def main():
    # quadruple precision
    mp.mp.prec = 113

    mats = Random_1s_noncommuting_MatrixGenerator(dim=8, seed=2).generate(3)
    t = 1
    error1 = []
    error2 = []
    ms = list(2**i for i in range(20))

    for m in ms:
        error1 += [norm_mpmath(ExactMatrixExponential()(t, mats) - TrotterFirstOrder(m)(t, mats), use_mpmath=True, p=2)]
        error2 += [norm_mpmath(ExactMatrixExponential()(t, mats) - TrotterSecondOrder(m)(t, mats), use_mpmath=True, p=2)]

    colors = color_palette("crest", 2)
    plt.plot(ms, error1, 'o', label="Lie-Trotter $S_1$", color=colors[0])
    plt.plot(ms, error2, 'o', label="Trotter-Suzuki $S_2$", color=colors[1])
    plt.plot(ms, [error1[0] / m for m in ms], 'k--', label=r"$\mathcal{O}(m^{-1})$")
    plt.plot(ms, [error2[0] / m**2 for m in ms], 'k--', label=r"$\mathcal{O}(m^{-2})$")
    plt.xscale("log", base=2)
    plt.xlabel("$m$")
    plt.yscale("log")
    plt.ylabel(r'$\epsilon$')
    plt.legend(bbox_to_anchor=(0.01, 0.01), loc='lower left')

    savefig("product_formula_convergence")


if __name__ == '__main__':
    main()
