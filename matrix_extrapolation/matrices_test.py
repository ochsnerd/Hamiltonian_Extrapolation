import numpy as np
import mpmath as mp

from .matrices import (Matrix,
                       allclose_mpmath, norm_mpmath, kron_mpmath,
                       commutator, max_commutator_norm, sum_commutator_norm,
                       nocommute)


# no tests for allclose_mpmath since it just forwards to np.allclose


def test_norm_mpmath():
    A = mp.zeros(10, 10)
    B = mp.ones(10, 10)
    B[1, 0] *= 1j

    assert norm_mpmath(A) == 0, ""
    assert norm_mpmath(A, use_mpmath=True) == 0, ""
    assert norm_mpmath(A, use_mpmath=True, p=2) == 0, ""

    assert np.isclose(norm_mpmath(B), np.linalg.norm(np.array(B.tolist(), dtype=complex))), ""
    assert np.isclose(float(norm_mpmath(B, use_mpmath=True)), float(mp.mnorm(B))), ""
    assert np.isclose(float(norm_mpmath(B, use_mpmath=True, p=2)),
                      np.linalg.norm(np.array(B.tolist(), dtype=complex), ord=2)), ""


def test_kron_mpmath():
    b = mp.ones(5) * mp.pi

    assert kron_mpmath(mp.eye(1), b) == b, ""
    assert kron_mpmath(b, mp.eye(1)) == b, ""

    c = mp.randmatrix(11, 5)

    assert (kron_mpmath(b, c) ==
            Matrix(np.kron(np.array(b.tolist()), np.array(c.tolist())))
            ), ""


def test_commutator():
    N = 10
    A = Matrix([list(range(N)) for _ in range(N)])
    B = mp.ones(N)

    assert allclose_mpmath(commutator(A, mp.eye(N)), mp.zeros(N)), ""
    assert allclose_mpmath(commutator(A, B), A @ B - B @ A), ""


def test_max_commutator_norm():
    A = mp.eye(2)
    B = mp.ones(2)
    C = mp.matrix([[1, 2], [3, 4]])
    D = mp.matrix([[5, 6], [7, 8]])

    assert np.isclose(max_commutator_norm([A, B]), 0), ""
    assert np.isclose(max_commutator_norm([D, C, B, A]), 17.8885438), ""
    assert np.isclose(max_commutator_norm([A, B, C, D], ord=1), 16), ""


def test_sum_commutator_norm():
    A = mp.eye(2)
    B = mp.ones(2)
    C = mp.matrix([[1, 2], [3, 4]])
    D = mp.matrix([[5, 6], [7, 8]])

    assert np.isclose(sum_commutator_norm([A, B]), 0), ""
    assert np.isclose(sum_commutator_norm([D, C, B, A]), 26.8328), ""
    assert np.isclose(sum_commutator_norm([A, B, C, D], ord=1), 24), ""


def test_nocommute():
    N = 10
    A = Matrix([list(range(N)) for _ in range(N)])
    B = mp.ones(N)

    assert nocommute([A, B]), ""
    assert not nocommute([A, B, mp.eye(N)]), ""
