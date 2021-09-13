from itertools import repeat
from typing import Tuple, Iterable, Sequence, Union

import sympy as sp

from .types import PS_Symbol, PS_Word, PS_Subterm
from .extrapolation_weights import ExtrapolationWeights, ExtrapolationWeightsSymmetric


def ps_words(order: int, length: int, allowed_symbols: Sequence[PS_Symbol]) \
        -> Iterable[PS_Word]:
    """Generate all words of length with order from allowed_symbols.

    Note that because the K_i don't commute, the ordering of the Symbols in the
    words is relevant.

    Combinatorially, these are the allowed_symbols-restricted compositions of
    order order with length length.

    For a symmetric product formula (only odd terms in BCH expansion),
    the restriced_symbols are all odd numbers (can be truncated at order).

    The number of terms generated is given by the number of compositions
    given for the setup, see the docstring of restriced_k_compositions.
    However, this function will be used in a context where all possible
    (restricted) compositions will be enumerated exactly once.

    >>> for W in PS_words(4, 6, [1, 3, 5]):
    >>>     print(W)
    (1, 1, 1, 3)
    (1, 1, 3, 1)
    (1, 3, 1, 1)
    (3, 1, 1, 1)
    """
    yield from restricted_k_compositions(order, length, allowed_symbols)


def restricted_k_compositions(n: int, k: int, A: Sequence[int]) \
        -> Iterable[Tuple[int, ...]]:
    """Generate all A-restricted k-compositions of n.

    A compositon of a positive integer n is a way of writing n as the sum of a
    sequence of positive integers.
    An A-restricted composition of n is an ordered collection of one or
    more elements in A whose sum is n.
    A k-composition is a composition restriced to exactly k terms.

    A is assumed to be ordered and contain no duplicates.

    The generated values are yielded in lexicographical ordering.

    The number of terms generated is given by the number of compositions
    given for the setup.
    When no length is prescribed, the number of unrestricted compositions of n
    is given by 2**(n-1).
    There are F_{n} odd-restricted compositions of n > 1, where F_i is the
    i-th Fibonacci number (OEIS A000045).
    The number of k-compositions is given by the binomial coefficient
    binomial(n - 1, k - 1).

    >>> for x in restriced_k_compositions(6, 2, [1, 3, 5]):
    >>>     print(x)
    (1, 5)
    (3, 3)
    (5, 1)

    >>> for x in restriced_k_compositions(6, 4, [1, 3, 5]):
    >>>     print(x)
    (1, 1, 1, 3)
    (1, 1, 3, 1)
    (1, 3, 1, 1)
    (3, 1, 1, 1)

    >>> for x in restriced_k_compositions(6, 6, [1, 3, 5]):
    >>>     print(x)
    (1, 1, 1, 1, 1, 1)

    Kuth's [Algorithm T] in Knuth2011 can be used to generated unrestriced
    k-combinations.
    I haven't found a way to adapt it to restricted k-combinations.

    References:
    https://en.wikipedia.org/wiki/Composition_(combinatorics)

    Knuth, D.E., 2011.
    The art of computer programming, volume 4A: combinatorial algorithms, part 1.
    Pearson Education India.
    """
    assert n >= 0 and k >= 0
    assert all(a > 0 for a in A)
    assert list(A) == sorted(set(A))

    def helper(target, taken, len_taken):
        assert len(taken) == len_taken
        if target < 0 or len_taken > k:
            return
        if target == 0 and len_taken == k:
            yield tuple(taken)
        for a in A:
            yield from helper(target - a, taken + [a], len_taken + 1)

    A = tuple(a for a in A if a <= n)
    yield from helper(n, [], 0)


def ps_prefactor_symmetric(k: int,
                           r: int,
                           m: Sequence[int],
                           a: ExtrapolationWeightsSymmetric) \
        -> sp.Rational:
    """Compute the prefactor for a (k,r)-word given an (m, a) symmetric extrapolation scheme."""
    assert len(m) == len(a)
    return sp.Rational(sum(aj * mj**(-2*r) for aj, mj in zip(a, m)),
                       sp.factorial(k - 2*r))


def ps_prefactor(k: int,
                 r: int,
                 m: Sequence[int],
                 a: ExtrapolationWeights) \
        -> sp.Rational:
    """Compute the prefactor for a (k,r)-word given an (m, a) extrapolation scheme."""
    assert len(m) == len(a)
    return sp.Rational(sum(aj * mj**(-r) for aj, mj in zip(a, m)),
                       sp.factorial(k - r))


def compute_ps_terms(order: int,
                     m: Sequence[int],
                     a: Union[ExtrapolationWeights, ExtrapolationWeightsSymmetric],
                     BCH_terms: Sequence[PS_Symbol]) \
        -> Iterable[PS_Subterm]:
    """Compute the term of order order in the power series expansion of \
    the extrapolation-error.

    For a product formula with BCH-expansion given by exp(sum x^n/m^(n-1)*BCH_terms[n]),
    compute the order-th order term of the power-series expansion of the extrapolation
    error.

    Will generate 2**order terms if BCH_terms is 1,2,3,... .
    See the docstring of ps_words for more.

    If we only use m=[1] extrapolation steps, we compute the order
    term in the error expansion of a single Trotter step:

    >>> for factor, word in compute_ps_terms(5, [1], (1, 3, 5)):
    >>>     print(factor, "*", word)
    1/6 * (1, 1, 3)
    1/6 * (1, 3, 1)
    1/6 * (3, 1, 1)
    1 * (5,)
    """
    k = order
    l = len(m)

    if isinstance(a, ExtrapolationWeightsSymmetric):
        for r in range(l, (k - 1) // 2 + 1):
            pref = ps_prefactor_symmetric(k, r, m, a)
            yield from zip(repeat(pref), ps_words(k, k - 2*r, BCH_terms))
    else:
        for r in range(l, k):
            pref = ps_prefactor(k, r, m, a)
            yield from zip(repeat(pref), ps_words(k, k - r, BCH_terms))
