import sympy as sp

from typing import Sequence, Union


class ExtrapolationWeightsSymmetric(Sequence):
    """Cached extrapolation-weights a, defined as

    a_j = prod_q m_j^2 / (m_j^2 - m_q^2)
    """
    def __init__(self, m: Sequence[int]) -> None:
        assert len(m) == len(set(m))
        assert all(m_j > 0 for m_j in m)

        m2 = [m_j**2 for m_j in m]
        self.a = []
        for m_j2 in m2:
            self.a += [sp.prod(sp.Rational(m_j2, m_j2 - m_q2)
                               for m_q2 in m2 if m_q2 != m_j2)]

    def __getitem__(self, j: Union[int, slice]) -> sp.Rational:
        return self.a[j]

    def __len__(self):
        return len(self.a)


class ExtrapolationWeights(Sequence):
    """Cached extrapolation-weights a, defined as

    a_j = prod_q m_j / (m_j - m_q)
    """
    def __init__(self, m: Sequence[int]) -> None:
        assert len(m) == len(set(m))
        assert all(m_j > 0 for m_j in m)

        self.a = []
        for m_j in m:
            self.a += [sp.prod(sp.Rational(m_j, m_j - m_q)
                               for m_q in m if m_q != m_j)]

    def __getitem__(self, j: Union[int, slice]) -> sp.Rational:
        return self.a[j]

    def __len__(self):
        return len(self.a)
