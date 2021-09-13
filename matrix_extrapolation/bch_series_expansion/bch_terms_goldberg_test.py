import os, tempfile

import sympy as sp

from .bch_terms_goldberg import (BCH_words_of_order, compute_m_mp_s,
                                 GPolynomials, compute_Goldberg_coefficient,
                                 prepend_coeff_to_word, integrate_word,
                                 word_to_operator, term_to_operator, BCH_term)
from .product_formula import S2
from .operator import strOperator


def test_BCH_words_of_order():
    assert tuple(BCH_words_of_order(0, [1, 2, 3])) == ((),)
    assert tuple(BCH_words_of_order(2, [1, 2, 3], {1, 2, 3})) == ()

    assert tuple(BCH_words_of_order(2, [1])) == ((1, 1),)
    assert tuple(BCH_words_of_order(2, [1, 2])) == ((1, 1), (1, 2), (2, 1), (2, 2),)
    assert tuple(BCH_words_of_order(2, [1, 2], {2})) == ((1, 1), (2, 1))


def test_compute_m_mp_s():
    assert compute_m_mp_s(()) == (0, 0, ())
    assert compute_m_mp_s((1,)) == (1, 0, (1,))
    assert compute_m_mp_s((1,1,2,3,2,2)) == (4, 2, (2, 1, 1, 2))


def test_GPolynomials():
    G = GPolynomials()
    t = G.t
    assert G[2] == sp.poly(t - sp.Rational(1, 2))
    # [G[1], G[3]]
    assert G[1:4:2] == [sp.poly(1, t, domain='QQ'), sp.poly(t**2 - t + sp.Rational(1, 6))]


def test_compute_Goldberg_coefficient():
    assert compute_Goldberg_coefficient(0, 0, ()) == sp.Rational(2, 1)
    assert compute_Goldberg_coefficient(1, 0, (1,)) == sp.Rational(0, 1)
    assert compute_Goldberg_coefficient(3, 1, (1, 1, 2)) == sp.Rational(-1, 60), "Example (3.4) in Kobayashi1998"


def test_prepend_coeff_to_word():
    assert prepend_coeff_to_word(()) == (sp.Rational(2,1), ())
    assert prepend_coeff_to_word((1,)) == (sp.Rational(0), (1,))
    assert prepend_coeff_to_word((1,3,2,2)) == (sp.Rational(-1, 60), (1,3,2,2)), "Example (3.4) in Kobayashi1998"


def test_integrate_word():
    assert integrate_word((0, ()), (1,)) == (0, (1,))
    assert integrate_word((1, (1,)), (1, )) == (.5, (1, 1))

    assert integrate_word((2, ()), (1, 3)) == (2, (1,)), (
        "First order term for S2 gives 2A (integration constant B comes later)")
    assert (integrate_word((sp.Rational(-1, 60), (1,3,2,2)), (1, 3)) ==
            (sp.Rational(-1, 180), (1,3,2,2,1))), (
        "Example (3.4) in Kobayashi1998")


def test_word_to_operator():
    m = {1: strOperator("A"), 2: strOperator("B"), 3: strOperator("C")}

    assert str(word_to_operator((), m)) == "{0}"
    assert str(word_to_operator((1,), m)) == "A"
    assert str(word_to_operator((1,2), m)) == "[A, B]"
    assert str(word_to_operator((1,2,3), m)) == "[A, [B, C]]"


def test_term_to_operator():
    m = {1: strOperator("A"), 2: strOperator("B"), 3: strOperator("C")}
    assert str(term_to_operator((0, ()), m)) == "{0}"
    assert str(term_to_operator((1, (1,)), m)) == "1 * (A)"
    assert str(term_to_operator((2, (1, 2)), m)) == "2 * ([A, B])"
    assert str(term_to_operator((4, (1, 2, 3)), m)) == "4 * ([A, [B, C]])"


def test_BCH_term():
    s2 = S2((strOperator("2A"), strOperator("B")), "")

    # set $DATA_DIR to temp directory to force recomputation
    temp = tempfile.TemporaryDirectory()
    env = dict(os.environ)
    try:
        os.environ['DATA_DIR'] = temp.name

        assert str(BCH_term(s2, 0)) == "{0}", "string represenation of zero operator"
        # the strOp-'arithmetic' is becoming increasingly awkward
        assert str(BCH_term(s2, 1)) == "2 * (1/2 * (2A)) + 2 * (B) + -1 * (B)", (
            "Example (3.6) in Kobayashi1998")
        assert str(BCH_term(s2, 2)) == "{0}", (
            "S2 is symmetric")
        assert str(BCH_term(s2, 3)) == ("1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) "
                                        "+ 1/6 * ([B, [B, 1/2 * (2A)]]) "
                                        "+ 1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) + {0}"), (
                                            "Example (3.6) in Kobayashi1998")
    finally:
        os.environ.clear()
        os.environ.update(env)
