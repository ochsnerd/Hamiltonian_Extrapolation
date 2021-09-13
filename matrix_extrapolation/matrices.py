import numpy as np

from typing import Union, List
from random import random
from itertools import combinations

from mpmath import (matrix, zeros, eye, eigh,
                    sqrt,
                    mnorm, norm,
                    conj)


Matrix = matrix


def allclose_mpmath(a: Union[Matrix, np.ndarray],
                    b: Union[Matrix, np.ndarray],
                    *args, **kwargs) -> bool:
    """Convert a, b to ndarrays and return np.allclose(a, b)."""
    if isinstance(a, Matrix):
        a = np.array(a.tolist(), dtype=complex)
    if isinstance(b, Matrix):
        b = np.array(b.tolist(), dtype=complex)

    return np.allclose(a, b, *args, **kwargs)


def norm_mpmath(a: Union[Matrix, np.ndarray],
                use_mpmath: bool = False,
                *args, **kwargs) -> float:
    """Based on use_mpmath, compute either mp.mnorm(a) or np.linalg.norm(a).

    In both cases, args and kwargs are forwarded to the respective norm function.

    In the case of use_mpmath==False, a is converted into an ndarray
    with dtype=complex, possibly resulting in a loss of accuracy.

    ATTENTION: The default numpy-norm for matrices is Frobenius,
    while for mpmath it is the 1-norm.

    p='f' gives the Frobenius norm in mpmath.
    """
    if use_mpmath:
        if isinstance(a, np.ndarray):
            a = Matrix(a)
        if kwargs.get('p') == 2:
            # mpmath doesn't implement the spectral norm,
            # so compute it by hand.
            # https://mathworld.wolfram.com/SpectralNorm.html
            a_dagger = a.T
            for i in range(a_dagger.rows):
                for j in range(a_dagger.cols):
                    a_dagger[i, j] = conj(a_dagger[i, j])

            return sqrt(norm(eigh(a_dagger * a, eigvals_only=True), p='inf'))

        return mnorm(a, *args, **kwargs)

    if isinstance(a, Matrix):
        a = np.array(a.tolist(), dtype=complex)

    return np.linalg.norm(a, *args, **kwargs)


def kron_mpmath(a: Matrix, b: Matrix) -> Matrix:
    """Compute the Kronecker product.

    kron_mpmath is equivalent to np.kron, but works on
    Matrix instead of ndarray and does not convert
    Matrices to ndarrays.
    """
    ar, ac = a.rows, a.cols
    br, bc = b.rows, b.cols
    c = zeros(ar * br, ac * bc)
    for i in range(ar):
        for j in range(ac):
            c_ = a[i, j] * b
            for ii in range(br):
                for jj in range(bc):
                    c[i*br+ii, j*bc+jj] = c_[ii, jj]
    return c


def commutator(a: Matrix, b: Matrix) -> Matrix:
    """Compute [a,b] = ab - ba"""
    return a @ b - b @ a


def max_commutator_norm(mats: List[Matrix], ord=None) -> float:
    """Return the largest norm of the commutators from pairs among mats.

    ord is given to numpy.linalg.norm and determines which norm
    to use.
    """
    return max(map(lambda pair: norm_mpmath(commutator(*pair), ord=ord),
                   combinations(mats, 2)))


def sum_commutator_norm(mats: List[Matrix], ord=None) -> float:
    """Return the sum of the norms of the commutators from pairs among mats.

    ord is given to numpy.linalg.norm and determines which norm
    to use.
    """
    return sum(map(lambda pair: norm_mpmath(commutator(*pair), ord=ord),
                   combinations(mats, 2)))


def nocommute(mats: List[Matrix]) -> bool:
    """Return True if no two matrices in mats commute."""
    zero = zeros(mats[0].rows)
    # generate all unique pairs up to ordering, excluding pairs
    # of the same matrix
    for A, B in combinations(mats, 2):
        # using the norm here is very similar in terms of speed
        if allclose_mpmath(commutator(A, B), zero):
            return False

    return True


class Pauli:
    """Static class to hold Paulis

    Usage:
    >>> commutator(Pauli.X, Pauli.Y) - 2j * Pauli.Z
    matrix([[0,0], [0,0]])

    >>> Pauli[1] == Pauli.X
    True
    """
    X = Matrix([[0,   1], [1,  0]])
    Y = Matrix([[0, -1j], [1j, 0]])
    Z = Matrix([[1,   0], [0, -1]])

    def __class_getitem__(cls, i):
        return [eye(2), cls.X, cls.Y, cls.Z][i]
