import numpy as np

from mpmath import expm as mp_expm
from scipy.linalg import expm as scipy_expm

from .matrices import allclose_mpmath
from .product_formula import ExactMatrixExponential, TrotterFirstOrder, TrotterSecondOrder, Matrix


def test_ExactMatrixExponential():
    t = np.pi
    mats = [Matrix(0.5 * np.identity(2)), Matrix([[0, 1], [1, 0]])]

    assert allclose_mpmath(ExactMatrixExponential()(0, mats),
                           np.identity(2)), "time=0 is identity"
    assert allclose_mpmath(ExactMatrixExponential()(t, mats),
                           mp_expm(-1j * (mats[0] + mats[1]) * t)), ""


def test_TrotterFirstOrder():
    t = 0.5
    mats = [Matrix([[0, 0], [1, 0]]), Matrix([[0, 1], [1, 0]]), Matrix([[0, 1], [0, 0]])]

    T = TrotterFirstOrder()
    T.set_steps(0)
    assert allclose_mpmath(T(t, mats),
                           np.identity(2)), ("no steps returns identity")
    T.set_steps(1)
    assert allclose_mpmath(T(t, mats),
                           mp_expm(-1j * mats[0] * t) @ mp_expm(-1j * mats[1] * t) @ mp_expm(-1j * mats[2] * t)), (
                               "one step is just multiplication")
    T.set_steps(2)
    assert allclose_mpmath(T(t, mats),
                           (mp_expm(-.5j * mats[0] * t) @
                            mp_expm(-.5j * mats[1] * t) @
                            mp_expm(-.5j * mats[2] * t) @
                            mp_expm(-.5j * mats[0] * t) @
                            mp_expm(-.5j * mats[1] * t) @
                            mp_expm(-.5j * mats[2] * t))), ""

    mats2 = [np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 0]]), np.array([[0, 1], [0, 0]])]
    T.set_steps(1)
    assert np.allclose(T(t, mats2),
                       scipy_expm(-1j * mats2[0] * t) @ scipy_expm(-1j * mats2[1] * t) @ scipy_expm(-1j * mats2[2] * t)), (
                           "one step is just multiplication")
    T.set_steps(2)
    assert np.allclose(T(t, mats2),
                       (scipy_expm(-.5j * mats2[0] * t) @
                        scipy_expm(-.5j * mats2[1] * t) @
                        scipy_expm(-.5j * mats2[2] * t) @
                        scipy_expm(-.5j * mats2[0] * t) @
                        scipy_expm(-.5j * mats2[1] * t) @
                        scipy_expm(-.5j * mats2[2] * t))), ""


def test_TrotterSecondOrder():
    t = 0.5
    mats = [Matrix([[0, 0], [1, 0]]), Matrix([[0, 1], [1, 0]]), Matrix([[0, 1], [0, 0]])]

    T = TrotterSecondOrder()
    T.set_steps(0)
    assert allclose_mpmath(T(t, mats),
                           np.identity(2)), (
                               "no steps returns identiy")
    T.set_steps(1)
    assert allclose_mpmath(T(t, mats),
                           (mp_expm(-.5j * mats[0] * t) @
                            mp_expm(-.5j * mats[1] * t) @
                            mp_expm(-.5j * mats[2] * t) @
                            mp_expm(-.5j * mats[2] * t) @
                            mp_expm(-.5j * mats[1] * t) @
                            mp_expm(-.5j * mats[0] * t)
                            )), ""

    T.set_steps(2)
    assert allclose_mpmath(T(t, mats),
                           (mp_expm(-.25j * mats[0] * t) @
                            mp_expm(-.25j * mats[1] * t) @
                            mp_expm(-.25j * mats[2] * t) @
                            mp_expm(-.25j * mats[2] * t) @
                            mp_expm(-.25j * mats[1] * t) @
                            mp_expm(-.25j * mats[0] * t) @
                            mp_expm(-.25j * mats[0] * t) @
                            mp_expm(-.25j * mats[1] * t) @
                            mp_expm(-.25j * mats[2] * t) @
                            mp_expm(-.25j * mats[2] * t) @
                            mp_expm(-.25j * mats[1] * t) @
                            mp_expm(-.25j * mats[0] * t)
                            )), ""

    mats2 = [np.array([[0, 0], [1, 0]]), np.array([[0, 1], [1, 0]]), np.array([[0, 1], [0, 0]])]
    T.set_steps(1)
    assert np.allclose(T(t, mats2),
                       (scipy_expm(-.5j * mats2[0] * t) @
                        scipy_expm(-.5j * mats2[1] * t) @
                        scipy_expm(-.5j * mats2[2] * t) @
                        scipy_expm(-.5j * mats2[2] * t) @
                        scipy_expm(-.5j * mats2[1] * t) @
                        scipy_expm(-.5j * mats2[0] * t)
                        )), ""

    T.set_steps(2)
    assert np.allclose(T(t, mats2),
                       (scipy_expm(-.25j * mats2[0] * t) @
                        scipy_expm(-.25j * mats2[1] * t) @
                        scipy_expm(-.25j * mats2[2] * t) @
                        scipy_expm(-.25j * mats2[2] * t) @
                        scipy_expm(-.25j * mats2[1] * t) @
                        scipy_expm(-.25j * mats2[0] * t) @
                        scipy_expm(-.25j * mats2[0] * t) @
                        scipy_expm(-.25j * mats2[1] * t) @
                        scipy_expm(-.25j * mats2[2] * t) @
                        scipy_expm(-.25j * mats2[2] * t) @
                        scipy_expm(-.25j * mats2[1] * t) @
                        scipy_expm(-.25j * mats2[0] * t)
                        )), ""
