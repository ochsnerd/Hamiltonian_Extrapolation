from math import prod
from typing import Sequence, Mapping

from .bch_terms_goldberg import compute_BCH_terms as compute_BCH_terms_gb
from .bch_terms_casas import compute_BCH_terms as compute_BCH_terms_ca
from .operator import Operator
from .product_formula import ProductFormula, SymmetricProductFormula
from .extrapolation_weights import ExtrapolationWeights, ExtrapolationWeightsSymmetric
from .ps_terms import compute_ps_terms
from .types import PS_Subterm, PS_Symbol


def combine_BCH_ps(ps_term: PS_Subterm, BCH_terms: Mapping[PS_Symbol, Operator]) \
        -> Operator:
    """Compute the concrete value of a PS_Subterm as dictated by the BCH_terms"""
    return ps_term[0] * prod(BCH_terms[K_i] for K_i in ps_term[1])


def compute_extrapolation_error_term(
        order: int,
        product_formula: ProductFormula,
        extrapolation_steps: Sequence[int],
        goldberg: bool = True) -> Operator:

    if isinstance(product_formula, SymmetricProductFormula):
        extrapolation_weights = ExtrapolationWeightsSymmetric(extrapolation_steps)
    else:
        extrapolation_weights = ExtrapolationWeights(extrapolation_steps)

    if goldberg:
        BCH_terms_concrete = compute_BCH_terms_gb(product_formula, order)
    else:
        BCH_terms_concrete = compute_BCH_terms_ca(product_formula, order)

    BCH_terms_abstract = product_formula.ps_symbols(order)
    BCH_mapping = dict(zip(BCH_terms_abstract, BCH_terms_concrete))

    return sum(combine_BCH_ps(W, BCH_mapping)
               for W in compute_ps_terms(order,
                                         extrapolation_steps,
                                         extrapolation_weights,
                                         BCH_terms_abstract))
