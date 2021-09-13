import os

import matplotlib.pyplot as plt
# import tikzplotlib


def savefig(name):
    # for demonstration: just show figure
    plt.show()
    return
    try:
        fig_dir = os.environ['FIG_DIR']
    except KeyError:
        print("FIG_DIR not set, using ~/Documents/")
        fig_dir = os.path.join(os.environ['HOME'], "Documents")

    plt.tight_layout()

    full_path = os.path.join(fig_dir, name + '.tex')
    # tikzplotlib.clean_figure()
    tikzplotlib.save(full_path, axis_height=r'\figureheight', axis_width=r'\figurewidth')
    print("saved", full_path)


"""
Create Hamiltonian terms for a d-dimensional Ising lattice:

H = H_Z + H_X
H_Z = sum_{nearest neighbors i<j} Z_i Z_j
H_X = sum_i X_i

interaction weights or field strength
"""
from typing import Sequence

import numpy as np

from qiskit.quantum_info import Pauli


class IsingLatticeHamiltonian:
    # open chain
    def __init__(self, shape: Sequence[int]) -> None:
        assert all(s >= 0 for s in shape)
        self.shape = tuple(shape)
        self.n_dims = len(shape)
        self.n_sites = np.prod(self.shape)

    def transverse_field_term_paulis(self) -> Sequence[Pauli]:
        terms = []
        I = ["I"] * (self.n_sites - 1)
        for idx in range(self.n_sites):
            terms += [Pauli("".join(I[:idx] + ["X"] + I[idx:]))]
        return terms

    def interaction_term_paulis(self) -> Sequence[Pauli]:
        terms = []
        I = ["I"] * self.n_sites
        for idx in range(self.n_sites):
            coords = self._idx_to_coord(idx)
            for d in range(self.n_dims):
                neighbor_coords = np.copy(coords)
                # only check positive-direction neighbors
                # to avoid double-counting
                neighbor_coords[d] += 1
                try:
                    neighbor_idx = self._coord_to_idx(neighbor_coords)
                except ValueError:
                    # out of bounds
                    continue
                neighbor_interaction = I[:]
                neighbor_interaction[idx] = "Z"
                neighbor_interaction[neighbor_idx] = "Z"
                terms += [Pauli("".join(neighbor_interaction))]
        return terms

    def interaction_term_matrix(self) -> np.ndarray:
        return sum(p.to_matrix() for p in self.interaction_term_paulis())

    def transverse_field_term_matrix(self) -> np.ndarray:
        return sum(p.to_matrix() for p in self.transverse_field_term_paulis())

    def _coord_to_idx(self, coords: Sequence[int]) -> int:
        # no indexing from the back
        # numpy handles input errors nicely
        return np.ravel_multi_index(coords, self.shape)

    def _idx_to_coord(self, idx: int) -> Sequence[int]:
        # numpy handles input errors nicely
        return np.unravel_index(idx, self.shape)
