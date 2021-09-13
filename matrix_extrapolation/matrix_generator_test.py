from itertools import combinations

import numpy as np

from .matrix_generator import (Random_1s_noncommuting_MatrixGenerator,
                               Corner_1s_noncommuting_MatrixGenerator,
                               Heisenberg_XYZ_MatrixGenerator,
                               Heisenberg_EO_MatrixGenerator,
                               Poisson_Decomp_MatrixGenerator,
                               NormalizedMatrixGenerator)

from .matrices import Matrix, allclose_mpmath, norm_mpmath, nocommute, commutator


def test_Random_1s_noncommuting_matrix_generator():
    def col_one_sparse(mat):
        for i in range(mat.cols):
            if np.count_nonzero(mat[:, i]) > 1:
                return False
        return True

    g1 = Random_1s_noncommuting_MatrixGenerator(dim=5, seed=10)
    g2 = Random_1s_noncommuting_MatrixGenerator(dim=5, seed=2)

    assert not allclose_mpmath(g1.generate(1)[0], g2.generate(1)[0]), (
        "Different seeds need to create different sequences")
    assert allclose_mpmath(g1.generate(1)[0], g1.generate(1)[0]), (
        "Multiple seperate calls to generate the same sequence")
    assert all([col_one_sparse(m) for m in g1.generate(20)]), (
        "Generate 1 sparse matrices")

    g1.set_dim(4)
    mats = g1.generate(10)
    assert len(mats) == 10, ""
    assert all(m.rows == 4 and m.cols == 4 for m in mats), ""
    assert nocommute(mats), ""


def test_Corner_1s_noncommuting_MatrixGenerator():
    assert allclose_mpmath(Corner_1s_noncommuting_MatrixGenerator(3, step=1).generate(1)[0],
                           Matrix([[1, 0, 1],[0, 1, 0],[0, 0, 0]])), ""

    assert all(np.isclose(norm_mpmath(commutator(*Corner_1s_noncommuting_MatrixGenerator(5, step=i).generate(2))),
                          i) for i in range(1,5)), ""


def test_Poisson_Decomp_MatrixGenerator():
    mats = Poisson_Decomp_MatrixGenerator(8).generate(2)
    assert allclose_mpmath(mats[0],
                           np.array([[ 0.,-1., 0., 0., 0., 0., 0., 0.],
                                     [-1., 0., 0., 0., 0., 0., 0., 0.],
                                     [ 0., 0., 0.,-1., 0., 0., 0., 0.],
                                     [ 0., 0.,-1., 0., 0., 0., 0., 0.],
                                     [ 0., 0., 0., 0., 0.,-1., 0., 0.],
                                     [ 0., 0., 0., 0.,-1., 0., 0., 0.],
                                     [ 0., 0., 0., 0., 0., 0., 0.,-1.],
                                     [ 0., 0., 0., 0., 0., 0.,-1., 0.]])), ""
    assert allclose_mpmath(mats[1],
                           np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0.,-1., 0., 0., 0., 0., 0.],
                                     [0.,-1., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., 0.,-1., 0., 0., 0.],
                                     [0., 0., 0.,-1., 0., 0., 0., 0.],
                                     [0., 0., 0., 0., 0., 0.,-1., 0.],
                                     [0., 0., 0., 0., 0.,-1., 0., 0.],
                                     [0., 0., 0., 0., 0., 0., 0., 0.]])), ""

def test_Heisenberg_XYZ_MatrixGenerator():
    mats = Heisenberg_XYZ_MatrixGenerator(8).generate(3)
    assert allclose_mpmath(mats[0],
                           np.array([[0., 0., 0., 1., 0., 0., 1., 0.],
                                     [0., 0., 1., 0., 0., 0., 0., 1.],
                                     [0., 1., 0., 0., 1., 0., 0., 0.],
                                     [1., 0., 0., 0., 0., 1., 0., 0.],
                                     [0., 0., 1., 0., 0., 0., 0., 1.],
                                     [0., 0., 0., 1., 0., 0., 1., 0.],
                                     [1., 0., 0., 0., 0., 1., 0., 0.],
                                     [0., 1., 0., 0., 1., 0., 0., 0.]])), ""
    assert allclose_mpmath(mats[1],
                           np.array([[ 0.,  0.,  0., -1.,  0.,  0., -1., -0.],
                                     [ 0.,  0.,  1.,  0.,  0.,  0.,  0., -1.],
                                     [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
                                     [-1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                                     [ 0.,  0.,  1.,  0.,  0.,  0.,  0., -1.],
                                     [ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
                                     [-1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
                                     [-0., -1.,  0.,  0., -1.,  0.,  0.,  0.]])), ""
    assert allclose_mpmath(mats[2],
                           np.array([[2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0., -2.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0., -2.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.]])), ""


def test_Heisenberg_EO_MatrixGenerator():
    mats = Heisenberg_EO_MatrixGenerator(8).generate(2)
    assert allclose_mpmath(mats[0],
                           np.array([[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0., -1.,  0.,  2.,  0.,  0.,  0.],
                                     [0.,  0.,  0., -1.,  0.,  2.,  0.,  0.],
                                     [0.,  0.,  2.,  0., -1.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  2.,  0., -1.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])), ""
    assert allclose_mpmath(mats[1],
                           np.array([[1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                                     [0., -1.,  2.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  2., -1.,  0.,  0.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
                                     [0.,  0.,  0.,  0.,  0., -1.,  2.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  2., -1.,  0.],
                                     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])), ""


def test_NormalizedMatrixGenerator():
    d = NormalizedMatrixGenerator(Random_1s_noncommuting_MatrixGenerator(5))

    d.set_dim(6)
    assert d.generate(1)[0].rows == 6 and d.generate(1)[0].cols == 6, ""
    assert np.isclose(sum(norm_mpmath(commutator(A, B))
                          for A, B in combinations(d.generate(5), 2)),
                      1), "Normalized matrix commutators need to sum to 1"
