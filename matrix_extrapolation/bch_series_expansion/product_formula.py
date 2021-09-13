from abc import ABC, abstractmethod
from typing import Tuple, Mapping, Sequence

from sympy import Rational

from .types import BCH_Symbol, BCH_Subterm, PS_Symbol
from .operator import Operator, ZeroOperator


class ProductFormula(ABC):
    def __init__(self,
                 hamiltonian: Sequence[Operator],
                 hamiltonian_name: str) -> None:
        self.H_terms = hamiltonian
        self.J = len(self.H_terms)
        self.H_name = hamiltonian_name

    @abstractmethod
    def __str__(self) -> str:
        """Return a unique identifier of the ProductFormula (including \
        the Hamiltonian terms).
        """
        ...

    @abstractmethod
    def ps_symbols(self, up_to: int) -> Tuple[PS_Symbol, ...]:
        """Return the abstract terms that will appear in the BCH expansion.

        Which terms K_i appear in log(pf) = sum t^i K_i?

        Since any product formula will be at least first-order accurate,
        we can be sure that K_0 needs to be 0. A general product formula
        will have ps_symbol(N) be equivalent to
        tuple(range(1, N+1))

        For symmetric product formulae we know that only odd-order K_i
        appear, i.e.

        >>> pf.ps_symbol(N) == tuple(range(1, N+1, 2))
        True
        """
        # This function has a super bad name :(
        ...


class SymmetricProductFormula(ProductFormula):
    """Class to provide information specific to the choice of product-formula
    when computing the BCH expansion based on Goldberg coefficients.

    References:
    Hiroto Kobayashi, Naomichi Hatanoau, and Masuo Suzuki.
    Goldberg’s theorem and the Baker–Campbell–Hausdorff formula.
    Physica A: Statistical Mechanics and its Applications, 250(1):535–548, February 1998.
    """
    def __init__(self,
                 hamiltonian: Sequence[Operator],
                 hamiltonian_name: str) -> None:
        super().__init__(hamiltonian, hamiltonian_name)

    @abstractmethod
    def alphabet(self) -> Tuple[BCH_Symbol, ...]:
        """Return the symbols that make up the BCH expansion words.

        The number of symbols is given by the splitting in the product-
        formula and the number of terms in the Hamiltonian.

        As an example, the length of the alphabet for 2nd order Trotter
        with 2 Hamiltonian terms is 3:

        >>> secondorderTrotter((H_1, H_2), ...).alphabet()
        (1, 2, 3)
        """
        ...

    @abstractmethod
    def symbol_hamiltonian_map(self) -> Mapping[BCH_Symbol, Operator]:
        """Return a mapping between symbols and Hamiltonian-terms.

        Generally, pf.alphabet() == pf.symbol_hamiltonian_map().keys()
        (up to ordering) and len(pf.H_terms) <= len(pf.symbol_hamiltonian_map()).

        >>> secondorderTrotter((H_1, H_2), ...).symbol_hamiltonian_map()
        {1: .5*H_1, 2: H_2, 3: .5*H_1}
        """
        ...

    @abstractmethod
    def integration_symbols(self) -> Tuple[BCH_Symbol, ...]:
        """Return the symbols wrt. which the integration is done in Kobayashi1998.

        When computing BCH-expansion via Goldberg coefficients, there is an integration
        made with respect to some of the operators, see Kobayashi1998. Usually it's the
        symbols corresponding to the first term in the Hamiltonian.

        >>> secondorderTrotter((H_1, H_2), ...).integration_symbols()
        (1, 3)
        """
        ...

    @abstractmethod
    def integration_constant(self, order: int) -> Operator:
        """Return the integration constant from the integration in Kobayashi1998.

        The intergration constant is given by the BCH expansion of the terms
        that remain when the integration_symbols are set to 0. This corresponds
        to the BCH expansion of the same product formula with one less Hamiltonian
        term.

        The argument up_to indicates at which order (inclusive) to truncate
        the return value

        The return value has the format (scalar_muliple, sequence_of_symbols).

        >>> secondorderTrotter((H_1, H_2), ...).integration_constant()
        ((1, (2, )), )

        Would be cool: no up_to-argument, return generator that gives
        integration_constant in ascending order. Consumer can then just
        yield from generator until desired order is reached.
        """
        ...


class S1(ProductFormula):
    """Product formula corresponding to first-order Trotter.

    S_1(t) = exp(t A)exp(t B)
    """
    def __init__(self,
                 hamiltonian: Sequence[Operator],
                 hamiltonian_name: str) -> None:
        super().__init__(hamiltonian, hamiltonian_name)

    def ps_symbols(self, up_to: int) -> Tuple[PS_Symbol, ...]:
        """Return the abstract terms that will appear in the BCH expansion.

        Which terms K_i appear in log(pf) = sum t^i K_i.
        """
        return tuple(range(1, up_to + 1))

    def __str__(self) -> str:
        """Return a unique identifier of the ProductFormula (including \
        the Hamiltonian terms).
        """
        return f"S1_{self.H_name}_J={self.J}"


class S2(SymmetricProductFormula):
    """Product formula corresponding to second-order Trotter.

    S_2(t) = exp(t/2 A)exp(t B)exp(t/2 A)
    """
    def __init__(self,
                 hamiltonian: Sequence[Operator],
                 hamiltonian_name: str) -> None:
        super().__init__(hamiltonian, hamiltonian_name)

    def alphabet(self) -> Tuple[BCH_Symbol, ...]:
        """Return the symbols that make up the BCH expansion words.

        The only reason why the first symbol in the alphabet is 1
        (and not 0) is that it's more natural to exclude 0 when
        using integers as representations for symbols.

        >>> S2((H_1, H_2)).alphabet()
        (1, 2, 3)
        """
        alphabet_size = 2 * self.J - 1
        return tuple(range(1, alphabet_size + 1))

    def symbol_hamiltonian_map(self) -> Mapping[BCH_Symbol, Operator]:
        """Return a mapping between symbols and Hamiltonian-terms.

        Generally, pf.alphabet() == pf.symbol_hamiltonian_map().keys()
        (up to ordering) and len(pf.H_terms) <= len(pf.symbol_hamiltonian_map()).

        Note the definition of the terms used in the Goldberg-coefficients;
        scalar factors in the exponent are absorbed into the operator, such
        that S_2 = ... exp(A)exp(B)exp(A) ..., which means that all but the
        last operator are multiplied by 0.5. See Kobayashi1998.

        >>> S2((H_1, H_2), "").symbol_hamiltonian_map()
        {1: .5*H_1, 2: H_2, 3: .5*H_1}
        """
        alphabet = self.alphabet()
        h_terms = ([Rational(1, 2) * H_j for H_j in self.H_terms[:-1]] +
                   [self.H_terms[-1]] +
                   [Rational(1, 2) * H_j for H_j in reversed(self.H_terms[:-1])])
        return dict(zip(alphabet, h_terms))

    def integration_symbols(self) -> Tuple[BCH_Symbol, ...]:
        """Return the symbols wrt. which the integration is done in Kobayashi1998.

        When computing BCH-expansion via Goldberg coefficients, there is an integration
        made with respect to some of the operators, see Kobayashi1998. Usually it's the
        symbols corresponding to the first term in the Hamiltonian.

        >>> S2((H_1, H_2), "").integration_symbols()
        (1, 3)
        """
        return self.alphabet()[0], self.alphabet()[-1]

    def integration_constant(self, order: int) -> Operator:
        """Return the integration constant from the integration in Kobayashi1998.

        The intergration constant is given by the BCH expansion of the terms
        that remain when the integration_symbols are set to 0. This corresponds
        to the BCH expansion of the same product formula with one less Hamiltonian
        term.

        For 2nd order Trotter with 2 Hamiltonian terms, this is simply H_2.
        For higher-order product formulas or more terms, the expression
        is more involved and in general has an infinite number of terms.
        As an example, for 2nd order Trotter with 3 Hamiltonian terms,
        the integration constant is the BCH expansion of 2-term 2nd
        order Trotter.

        Instead of adding the integration constasnt at the abstract level,
        when the integration happens (like in Kobayashi1998 Eq. (3.6)),
        the integration constant is added after translating the words
        of BCH_Symbols to Operators. This way, the finished result of
        the preceding (one less Hamiltonian term) BCH term computation can be
        used.

        >>> S2((H_1, H_2), "").integration_constant(1)
        H_2
        """
        if order < 1:
            # no 0-th order integration constants
            return ZeroOperator()

        if self.J == 1:
            # this is a bit of a hack.
            # Kobayashi1998 is only valid for J >= 2.
            # For J=1, applying their result gives
            # exp(A) = exp(2A),
            # which comes from the fact that the Goldberg coefficient
            # is 2 for the empty string.
            # We cancel this by choosing an appropriate
            # integration constant.
            if order == 1:
                return (-1) * self.H_terms[0]
            return ZeroOperator()

        # pf with one less Hamiltonian term
        s2_p = S2(self.H_terms[1:], self.H_name)

        # this import are necessary because of the circular dependency
        # on the bch_terms functions when computing the integration_constant
        from . import bch_terms_goldberg

        return bch_terms_goldberg.cached_BCH_term(s2_p, order)

    def ps_symbols(self, up_to: int) -> Tuple[PS_Symbol, ...]:
        """Return the abstract terms that will appear in the BCH expansion.

        Which terms K_i appear in log(pf) = sum t^i K_i.

        Since S2 is symmetric, only odd-order terms appear,
        so:

        >>> S2([], "").ps_symbol(N) == tuple(range(1, N+1, 2))
        True
        """
        # This function has a super bad name :(
        return tuple(range(1, up_to + 1, 2))

    def __str__(self) -> str:
        """Return a unique identifier of the ProductFormula (including \
        the Hamiltonian terms).
        """
        return f"S2_{self.H_name}_J={self.J}"
