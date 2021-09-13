import os, tempfile, shutil
import time

import matplotlib.pyplot as plt
from seaborn import color_palette
import numpy as np

from matrix_extrapolation import (
    PauliOperator, S2,
    compute_extrapolation_error_term)

from ising_lattice import IsingLatticeHamiltonian

from util import savefig


def clear_dir(path):
    """https://stackoverflow.com/a/185941"""
    print("Deleting all contents of", path)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def complexity_full(n, N):
    # n: order
    # N: qubits
    M = 3
    return (M**(n-1) * np.e**(n/np.e) +
            (M**(n-1) * min(N**n, 4**(n*N)) +
             M**(2*(n-1)) * min(N**n, 4**N) +
             min(M**(2*(n-1)) * N**n, 4**(2*N)) +
             2**(n - 2) * min(M**(n * (n-1)) * N**(n*n), 4**(n*N)) +
             2**(2*n - 4) * min(M**(n*(n-1)) * N**(n*n), 4**N)) * N
            )


def main():
    temp = tempfile.TemporaryDirectory()
    env = dict(os.environ)

    recompute = True

    ks = [3, 5, 7, 9, 11]
    Ns = [3, 4, 5, 6, 7]
    # ks = [3, 5]
    # Ns = [3, 4]
    repeats = 5

    try:
        os.environ['DATA_DIR'] = temp.name

        for N, c in zip(Ns, color_palette("crest", len(Ns))):
            lattice_shape = (N, )
            H = IsingLatticeHamiltonian(lattice_shape)
            H_Z = PauliOperator(H.interaction_term_paulis())
            H_X = PauliOperator(H.transverse_field_term_paulis())

            pfs = []
            if recompute:
                for i in range(repeats):
                    pfs += [S2([H_Z, H_X], str(i) + str(N))]

                times = [[] for _ in ks]
                for i, k in enumerate(ks):
                    for pf in pfs:
                        if (N == 4 and k > 9) or (N == 5 and k > 7) or (N == 6 and k > 7) or (N == 7 and k > 5):
                            times[i] += [float('NaN')]
                            continue
                        start = time.perf_counter()
                        compute_extrapolation_error_term(k, pf, [1, 2])
                        stop = time.perf_counter()
                        times[i] += [stop - start]
                    clear_dir(temp.name)

                ts = np.array(times)
                with open('runtime_dat.npy', 'wb') as f:
                    np.save(f, ts)
            else:
                with open('runtime_dat.npy', 'rb') as f:
                    ts = np.load(f)
                
            mean = np.mean(ts, axis=1)
            lower_quartile = np.quantile(ts, 0.25, axis=1)
            upper_quartile = np.quantile(ts, 0.75, axis=1)
            print(f"{mean=}")
            print(f"{lower_quartile=}")
            print(f"{upper_quartile=}")

            plt.plot(ks, mean, 'o-', color=c, zorder=1,
                     label=f"$N = {N}$")
            # plt.errorbar(ks, mean, yerr=(mean - lower_quartile, upper_quartile - mean),
            #              color=c, fmt='', markersize=2, capsize=3)

            empirical = [N**(k) / N**(ks[0]) * mean[0] if not np.isnan(m) else float('NaN') for k, m in zip(ks, mean)]
            empirical = [N**(k) / N**(ks[0]) * mean[0] for k, m in zip(ks, mean)]
            predicted = [complexity_full(k, N) / complexity_full(ks[0], N) * mean[0] if not np.isnan(m) else float('NaN')
                               for k, m in zip(ks, mean)]
            if N == Ns[-1]:
                plt.plot(ks, empirical, 'k--', zorder=1, label=r"$\mathcal{O}\left(N^{n}\right)$")
                plt.plot(ks, predicted, 'k:', zorder=1, label=r"asymptotic")
            else:
                plt.plot(ks, empirical, 'k--', zorder=1)
                plt.plot(ks, predicted, 'k:', zorder=1)

        plt.legend(loc='upper left')
        plt.xlabel('$n$')
        plt.xticks(ks)
        plt.ylabel('time [$s$]')
        plt.yscale('log')
        plt.ylim([5e-2, 1e5])
        savefig("extrapolation_runtime")
    finally:
        os.environ.clear()
        os.environ.update(env)


if __name__ == '__main__':
    main()
