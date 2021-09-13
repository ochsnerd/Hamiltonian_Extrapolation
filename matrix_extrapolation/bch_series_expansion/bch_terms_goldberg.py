"""Compute terms in the BCH exansion of a product formula,
based on Goldberg coefficients as described in Kobayashi1998.

Hiroto Kobayashi, Naomichi Hatanoau, and Masuo Suzuki.
Goldberg’s theorem and the Baker–Campbell–Hausdorff formula.
Physica A: Statistical Mechanics and its Applications, 250(1):535–548, February 1998.
"""

import sympy as sp

from collections import Counter
from functools import lru_cache, reduce
from itertools import product, repeat
from math import prod
from typing import Iterable, Tuple, Set, Union, Mapping, Sequence

from .product_formula import SymmetricProductFormula
from .types import BCH_Symbol, BCH_Word, BCH_Subterm
from .operator import Operator, ZeroOperator
from .util import load_or_compute


def BCH_words_of_order(n: int,
                       alphabet: Iterable[BCH_Symbol],
                       disallowed_end: Set[BCH_Symbol] = set()) \
        -> Iterable[BCH_Word]:
    """Generate all possible combinations of symbols in alphabet of length n.

    Returns a generator yielding all possible combinations of length n from
    the alphabet.

    Filtering based on ideas from Kabayashi1998. For example combinations
    ending in a symbol in disallowed_end can be omitted (it would result
    in a commutator [A, A]).

    Without filtering, this function is equivalent to
    itertools.product(alphabet, repeat=n). This means in particular that
    the words are produced are ordered in the same way that the alphabet
    is ordered.

    The number of tuples generated (without filtering) is len(alphabet)**n.
    For 2nd order Trotter with J noncommuting Hamiltonian terms, the alphabet
    has size (2*J-1), so we generate (2*J-1)**n tuples of length n.

    All words ending in a BCH_Symbol in disallowed_set are filtered out.

    >>> for W in words_of_order(2, (1, 2, 3))
    >>>     print(W)
    (1, 1)
    (1, 2)
    (1, 3)
    (2, 1)
    (2, 2)
    (2, 3)
    (3, 1)
    (3, 2)
    (3, 3)

    >>> for W in words_of_order(3, (1, 2, 3), {1, 3})
    >>>     print(W)
    (1, 1, 2)
    (1, 2, 2)
    (1, 3, 2)
    (2, 1, 2)
    (2, 2, 2)
    (2, 3, 2)
    (3, 1, 2)
    (3, 2, 2)
    (3, 3, 2)
    """
    if n != 0:
        assert alphabet, "Require non-empty alphabet to generate words"

    all_words = product(alphabet, repeat=n)
    if disallowed_end and n > 0:
        # if n is 0, all_words is ().
        # we're sure there are no words with disallowed endings
        return filter(lambda W: not W[-1] in disallowed_end, all_words)
    return all_words


def prepend_coeff_to_word(W: BCH_Word) -> BCH_Subterm:
    """Prepend the Goldberg coefficent to word.

    Compute the coefficients as described in Kobayashi1998 (Corollary 1, Equation 2.7).

    The work per word is proportional to the length of the word.

    The actual computation (integral) is cached.

    >>> prepend_coeff_to_word((1, 3, 2, 2))
    (-1/60, (1, 3, 2, 2))
    """
    return compute_Goldberg_coefficient(*compute_m_mp_s(W)), W


def compute_m_mp_s(W: BCH_Word) -> Tuple[int, int, Tuple[int, ...]]:
    """Compute the variables m, m' and [s, ...] for a given word W.

    Based on the definitions in Theorem 1 in Kobayashi1998.

    m: number of distinct symbols, contracting
    adjacent identical symbols.
    (1,1,2,3,2,2) -> m=4

    m': number of symbol changes that increase the symbol value.
    (1,1,2,3,2,2) -> m' = 2

    s: tuple of exponents
    (1,1,2,3,2,2)  -> s = (2, 1, 1, 2)

    Returns a tuple (m, m', s)
    """
    if not W:
        return 0, 0, ()

    if len(W) == 1:
        return 1, 0, (1,)

    m = 1
    mp = 0
    s_list = [1]
    for X_curr, X_next in zip(W, W[1:]):
        if X_curr != X_next:
            m += 1
            s_list.append(0)
            if X_curr < X_next:
                mp += 1
        s_list[-1] += 1

    return m, mp, tuple(s_list)


@lru_cache(maxsize=1024)
def compute_Goldberg_coefficient(m: int, mp: int, s_seq: Sequence[int]) \
        -> sp.Rational:
    """Compute the coefficent of a word characterised by m, mp, s_seq.

    See Corollary 1 in Kobayashi1998.
    Uses sympy to compute the integral.
    Keeps the least recent 1024 results in cache.
    """
    assert len(s_seq) == m, ""
    assert m > mp or m == 0, ""

    # create global instance of G_polys to cache created polynomials
    global G_polys
    try:
        G_polys
    except NameError:
        G_polys = GPolynomials()

    t = G_polys.t
    if m == 0:
        # we get a rational function and not a polynomial as the integrand
        # > different sympy syntax
        # > cast resulting float to sympy.Rational

        # this is not the correct way to evaluate this integral.
        # however, the integral evaluated correctly does not converge, and the
        # result I obtain matches the result in the paper.
        return sp.Rational(sp.integrate((2*t - 1)*t**(-1)).evalf(subs={t: 1}))

    G_prod = prod(G_polys[s] for s in s_seq)
    prefactor = sp.poly((2*t - 1) * (t - 1)**mp * t**(m - mp - 1), t)

    Int = (prefactor * G_prod).integrate()(1)
    return (-1)**sum(s_seq) * Int

# polynomial multiplication: see eq (4.1.6) pg 287 in Hiptmair script

class GPolynomials:
    """Polynomials as defined by (2.4) in Kobayashi1998.

    Definition is extended to G_0 = 0, such that the
    0-based index-access lines up with the mathematical
    1-based indexing.
    """
    def __init__(self) -> None:
        self.t = sp.Symbol('t')
        self.G_list = [sp.poly(0, self.t, domain='QQ'),
                       sp.poly(1, self.t, domain='QQ')]

    def __getitem__(self, s: Union[int, slice]) -> sp.Poly:
        """Cached access to the s-th G-polynomial.

        Note that this returns G_s (and not G_{s+1}).

        Since the G-polynomials are recusively defined,
        there is no upper limit to s. This also means
        that indexing from the back is not possible.

        For practical reasons, slicing only works up to an endpoint of
        10000.
        """
        if isinstance(s, slice):
            if s.start is not None and s.start < 0:
                raise ValueError("Can't index infinite sequence from the back")
            if s.stop is None or s.stop < 0:
                raise ValueError("Can't index infinite sequence from the back")
            # if slice-end points beyond the end of the currently generated
            # list, generate up to endpoint
            M = 10000
            self[s.indices(M)[1]]
            return [self[i] for i in range(*s.indices(len(self.G_list)))]
        if s < 0:
            raise ValueError("Can't index infinite sequence from the back")

        while s >= len(self.G_list):
            # incrementally build up G_s until we can return the required one
            self.G_list.append(
                sp.poly(
                    (sp.diff(self.t * (self.t - 1) * self.G_list[-1])
                     / len(self.G_list))))
        return self.G_list[s]


def integrate_word(coeff_word_pair: BCH_Subterm,
                   integration_symbols: Sequence[BCH_Symbol]) \
        -> BCH_Subterm:
    """Perform 'trick' from Kobayashi1998 to get a sequence of symbols representing a
    commutator from a commutator operator.

    Integrate with respect to epsilon (indicated by the argument integration_symbol),
    so there is an integration factor to multiply with the scalar factor.
    The algorithm multiplies all integration_symbols with epsilon and then integrates
    with respect to epsilon. We can emulate this by counting the number of
    integration_symbols in a word and then multiplying it's scalar factor by the
    appropiate factor, i.e.
    occurences = collections.Counter(W)
    sp.Rational(1, sum(occurences[s] for s in integration_symbols) + 1)

    The integration constant is set to 0 (and has to be added later).

    Append integration_symbol[0] to get meaningful commutators.
    Assuming a mapping {1: X, 2: Y, 3: X},
    an input-word to this function is interpreted as:
    (1, 3, 2, 2)    -> [X, [X, [Y, [Y, .]]]]
    Assuming the integation_symbols are (1, 3), the output-word will look like and
    is interpreted as:
    (1, 3, 2, 2, 1) -> [X, [X, [Y, [Y, X]]]]

    The work per word is proportional to the length of the word.

    >>> integrate_word((-1/60, (1, 3, 2, 2)), (1, 3))
    (-1/180, (1, 3, 2, 2, 1))
    """
    c, W = coeff_word_pair
    
    # determine resulting factor from integrating  w.r.t
    # (a scalar factor of) symbols in integration_symbols.
    occurences = Counter(W)
    c /= 1 + sum(occurences[S] for S in integration_symbols)

    # this copies all elements in W - but the solution to that would
    # be to use a list for BCH_Word
    W += (integration_symbols[0],)

    return c, W


def word_to_operator(W: BCH_Word,
                     symbol_operator_map: Mapping[BCH_Symbol, Operator]) \
        -> Operator:
    """Translate word W to an operator by computing the commutators.

    Assumes that the Operator has a method Operator.commutator(other).

    >>> (word_to_operator((1, 3, 2, 2, 1), M) ==
    ...     comm(M[1], comm(M[3], comm(M[2], comm(M[2], M[1])))))
    True
    """
    if not W:
        return ZeroOperator()
    if len(W) == 1:
        return symbol_operator_map[W[0]]

    # translate word to operator, reverse order (commutators are right-nested)
    W_op = list(map(lambda S: symbol_operator_map[S], reversed(W)))
    # compute nested commutator from right to left
    return reduce(lambda C, O: O.commutator(C),
                  W_op[2:],
                  W_op[1].commutator(W_op[0]))


def term_to_operator(coeff_word_pair: BCH_Subterm,
                     symbol_operator_map: Mapping[BCH_Symbol, Operator]) \
        -> Operator:
    """Convert a word to an operator and multiply it by its coefficient."""
    c, W = coeff_word_pair
    return word_to_operator(W, symbol_operator_map) * c


def BCH_term(product_formula: SymmetricProductFormula,
             order: int) -> Operator:
    """Compute the term of order order in the BCH expansion of product_formula."""
    if order < 1:
        return ZeroOperator()
    alphabet = product_formula.alphabet()
    # int_symbols represent the same operator. int_symbols[0] is appended to all words
    # in the integration step, so words ending in one of them will be 0
    int_symbols = product_formula.integration_symbols()
    words = BCH_words_of_order(order - 1, alphabet, set(int_symbols))
    coeff_word_pairs = filter(lambda cW: cW[0] != 0,
                              map(prepend_coeff_to_word, words))

    integrated_words = map(integrate_word, coeff_word_pairs, repeat(int_symbols))

    s_to_H_op_map = product_formula.symbol_hamiltonian_map()

    terms_as_operators = map(term_to_operator, integrated_words, repeat(s_to_H_op_map))

    return (sum(terms_as_operators, start=ZeroOperator()) +
            product_formula.integration_constant(order))


def cached_BCH_term(product_formula: SymmetricProductFormula,
                    order: int) -> Operator:
    """Attempt to load a stored BCH_term for product_formula from disk.

    If it isn't there, compute it and store it.

    The directory that is searched is indicated in the
    environment-variable $DATA_DIR, with $HOME/Documents as a fallback."""
    return load_or_compute(lambda: BCH_term(product_formula, order),
                           f"{product_formula}_o={order}")


def compute_BCH_terms(product_formula: SymmetricProductFormula,
                      max_order: int) \
        -> Tuple[Operator]:
    """Load or compute BCH terms up to order max_order"""
    assert isinstance(product_formula, SymmetricProductFormula), (
        "Goldberg only works for symmmetric product formulas")
    terms = []
    # this loop assume that PS_Symbol is castable to int
    for o in product_formula.ps_symbols(max_order):
        terms += [cached_BCH_term(product_formula, int(o))]

    return tuple(terms)
