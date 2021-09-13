from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import product
from numbers import Complex
from typing import Union, Sequence, Mapping

import numpy as np
import sympy as sp

from qiskit.quantum_info import Pauli


class Operator(ABC):
    @abstractmethod
    def __add__(self, other: Operator) -> Operator:
        ...

    @classmethod
    @abstractmethod
    def _operator_mul(cls, A: Operator, B: Operator) -> Operator:
        ...

    @abstractmethod
    def _scalar_mul(self, scalar: Complex) -> Operator:
        ...

    @abstractmethod
    def commutator(self, other: Operator) -> Operator:
        ...

    def __radd__(self, other: Operator) -> Operator:
        # addition is commutative
        return self + other

    def __sub__(self, other: Operator) -> Operator:
        return self + (-1 * other)

    def __rsub__(self, other: Operator) -> Operator:
        return other + (-1 * self)

    def __mul__(self, other: Union[Complex, Operator]) -> Operator:
        if isinstance(other, Operator):
            return self._operator_mul(self, other)
        if isinstance(other, Complex):
            return self._scalar_mul(other)
        return NotImplemented

    def __rmul__(self, other: Union[Complex, Operator]) -> Operator:
        if isinstance(other, Operator):
            return self._operator_mul(other, self)
        if isinstance(other, Complex):
            return self._scalar_mul(other)
        return NotImplemented

    def __matmul__(self, other: Union[Complex, Operator]) -> Operator:
        return self * other

    def __rmatmul__(self, other: Union[Complex, Operator]) -> Operator:
        return other * self


class ZeroOperator(Operator):
    def __add__(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            return other
        return NotImplemented

    @classmethod
    def _operator_mul(cls, A: Operator, B: Operator) -> Operator:
        if isinstance(A, Operator) and isinstance(B, Operator):
            return cls()
        return NotImplemented

    def _scalar_mul(self, other: Operator) -> Operator:
        return self

    def commutator(self, other: Operator) -> Operator:
        if isinstance(other, Operator):
            return self
        return NotImplemented

    def __str__(self) -> str:
        return "{0}"


class strOperator(Operator):
    def __init__(self, a):
        self.a = a

    def __add__(self, other):
        return strOperator(f"{self} + {other}")

    @classmethod
    def _operator_mul(cls, A, B):
        return cls(f"({A}) * ({B})")

    def _scalar_mul(self, scalar):
        return strOperator(f"{scalar} * ({self})")

    def __str__(self):
        return str(self.a)

    def commutator(self, other):
        return strOperator(f"[{self}, {other}]")


class SympyOperator(Operator):
    def __init__(self, Op, s=sp.Rational(1, 1)):
        self.s = sp.sympify(s)
        try:
            if isinstance(Op, (sp.Symbol, sp.Pow, sp.Mul, sp.Add, Complex)):
                self.Op = Op
            elif isinstance(Op, str):
                self.Op = sp.Symbol(Op, commutative=False)
            else:
                raise TypeError
        except TypeError:
            raise RuntimeError(f"Got {type(Op)} instead of sympy.Symbol, "
                               "sympy Operation, numbers.Number or str.")

    def __add__(self, other):
        if isinstance(other, SympyOperator):
            # not doing any factoring here...
            return SympyOperator(self.s * self.Op + other.s * other.Op)
        if other == 0:
            return self
        return NotImplemented

    @classmethod
    def _operator_mul(cls, A, B):
        if isinstance(A, SympyOperator) and isinstance(B, SympyOperator):
            return cls(A.Op * B.Op, A.s * B.s)
        return NotImplemented

    def _scalar_mul(self, scalar):
        return SympyOperator(self.Op, self.s * scalar)

    def commutator(self, other):
        if isinstance(other, SympyOperator):
            # "fake" commutator by defining new Symbol
            s = '[' + str(self.Op) + ', ' + str(other.Op) + ']'
            return SympyOperator(s, self.s * other.s)
        return NotImplemented

    def __str__(self):
        return str(sp.expand(self.Op * self.s))

    def pprint(self):
        sp.pprint(sp.expand(self.Op * self.s))

    def latex(self):
        return sp.latex(sp.expand(self.Op * self.s))


class PauliOperator(Operator):
    def __init__(self, paulis: Sequence[Pauli],
                 scalar_muliples: Sequence[Complex]=None) -> None:
        if not isinstance(paulis, Sequence):
            # Since Pauli.len is defined and Pauli is iterable, having the argument pauli
            # be just a single Pauli (instead of a list with only one entry)
            # doesn't raise an error, but introduces a bug.
            # Make sure here that we're actually working with an iterable.
            paulis = [paulis]
        if scalar_muliples is None:
            scalar_muliples = [1 for _ in paulis]
        self.s_P = PauliOperator._dict_from_pauli_scalar_sequences(paulis, scalar_muliples)

    def __add__(self, other: PauliOperator) -> PauliOperator:
        if isinstance(other, Complex):
            if other == 0:
                return self
        if isinstance(other, PauliOperator):
            new_s_P = {**self.s_P, **other.s_P}
            # add scalar multiples that have been overwritten by
            # dict merge (don't use self._dict_from_pauli_scalar_sequences
            # because we don't need to extract phase)
            # Here, scalar multiples could become 0
            for p in self.s_P.keys() & other.s_P.keys():
                new_s_P[p] += self.s_P[p]
            return PauliOperator._from_dict(new_s_P)
        return NotImplemented

    @classmethod
    def _operator_mul(cls, A: PauliOperator, B: PauliOperator) -> PauliOperator:
        if isinstance(A, PauliOperator) and isinstance(B, PauliOperator):
            # Pa & Pb == Pb * Pa
            paulis, scalars = zip(*((BPj & APi, A.s_P[APi] * B.s_P[BPj])
                                    for APi, BPj in product(A.s_P, B.s_P)))
            return cls._from_dict(cls._dict_from_pauli_scalar_sequences(paulis, scalars))
        return NotImplemented

    def _scalar_mul(self, scalar: Complex) -> PauliOperator:
        return PauliOperator._from_dict({P_i: s_i * scalar
                                         for P_i, s_i in self.s_P.items()})

    def commutator(self, other: PauliOperator) -> PauliOperator:
        if isinstance(other, PauliOperator):
            return pauli_commutator(self, other)
        return NotImplemented

    def to_matrix(self) -> np.ndarray:
        if not self.s_P:
            # can't infer dimension, return 1D ndarray
            return np.array([0], dtype=complex)
        return sum(s_i * P_i.to_matrix() for P_i, s_i in self.s_P.items())

    @classmethod
    def from_PauliOp(cls, P: PauliOperator) -> PauliOperator:
        return cls._from_dict(deepcopy(P.s_P))

    @classmethod
    def _from_dict(cls, d: dict) -> PauliOperator:
        # ! d is not copied, make sure it is not changed !
        # no extracting of phase!
        new = cls([], [])
        new.s_P = d
        return new

    @classmethod
    def _dict_from_pauli_scalar_sequences(cls, paulis: Sequence[Pauli],
                                          scalar_muliples: Sequence[Complex]) \
            -> Mapping[Pauli, Complex]:
        assert len(paulis) == len(scalar_muliples)
        # paulis could contain duplicates
        new_s_P = defaultdict(complex)
        for phased_P, s in zip(paulis, scalar_muliples):
            phase = cls._phase_int_to_complex[phased_P.phase]
            phased_P.phase = 0
            new_s_P[phased_P] += phase * s
        return new_s_P

    _phase_int_to_complex = {0: 1, 1: -1j, 2: -1, 3: 1j}


@lru_cache
def pauli_commutator(A, B):
    return A * B - B * A
