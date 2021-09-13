import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from seaborn import color_palette

from util import savefig

from matrix_extrapolation import (RichardsonExtrapolation,
                                  TrotterSecondOrder,
                                  ExactMatrixExponential,
                                  norm_mpmath,
                                  Random_1s_noncommuting_MatrixGenerator)


def main():
    mp.mp.prec = 113

    mats = Random_1s_noncommuting_MatrixGenerator(dim=8, seed=2).generate(3)

    ts = np.logspace(-5, 1, base=2, num=20)
    ls = [1, 3, 5]
    mss = [list(range(1, l + 1)) for l in ls]
    errors = [[] for _ in ls]
    colors = color_palette("crest", len(ls))

    for t in ts:
        for errors_list, ms in zip(errors, mss):
            errors_list += [norm_mpmath(ExactMatrixExponential()(t, mats) -
                                        RichardsonExtrapolation(TrotterSecondOrder(), ms)(t, mats),
                                        use_mpmath=True, p=2)]

    plt.plot(ts, errors[0], 'o', label="Suzuki-Trotter $S_2 = M^1_{S_2}$", color=colors[0])
    for errors_list, color, l in zip(errors[1:], colors[1:], ls[1:]):
        plt.plot(ts, errors_list, 'o', label="Extrapolation on $" + str(l) + "$ points $M^" + str(l) + "_{S_2}$", color=color)

    for l, errors_list in zip(ls[1:], errors[1:]):
        exponent = 2*l+1
        plt.plot(ts, [errors_list[0] * (t / ts[0])**exponent for t in ts], 'k--')

    plt.plot(ts, [errors[0][0] * (t / ts[0])**3 for t in ts], 'k--', label=r"$\mathcal{O}(t^{2l + 1})$")
    plt.xscale("log", base=2)
    plt.xlabel("$t$")
    plt.yscale("log")
    plt.ylabel(r'$\epsilon$')
    plt.legend()

    savefig("extrapolation_convergence")


if __name__ == '__main__':
    main()
