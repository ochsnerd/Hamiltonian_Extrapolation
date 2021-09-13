import os, tempfile

from .combine import compute_extrapolation_error_term
from .operator import strOperator
from .product_formula import S2


def test_compute_extrapolation_error_term():
    # set $DATA_DIR to temp directory to force recomputation
    temp = tempfile.TemporaryDirectory()
    env = dict(os.environ)
    try:
        os.environ['DATA_DIR'] = temp.name

        s2 = S2((strOperator("2A"), strOperator("B")), "")
        assert str(compute_extrapolation_error_term(3, s2, [1])) == (
            "1 * (1 * (1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) "
            "+ 1/6 * ([B, [B, 1/2 * (2A)]]) + "
            "1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) + {0})) + 0"), "(24) in my notes"

        assert str(compute_extrapolation_error_term(4, s2, [1])) == (
            "1/2 * "
            "((1 * (2 * (1/2 * (2A)) + 2 * (B) + -1 * (B))) "  # (2A + B)
            "* (1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) + 1/6 * ([B, [B, 1/2 * (2A)]]) + 1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) + {0})"  # * (-1/12 [A, [A, B]] + 1/6 [B, [B, A]] - 1/12 [A, [A, B]])
            ") + 0 + "
            "1/2 * "
            "((1 * (1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) + 1/6 * ([B, [B, 1/2 * (2A)]]) + 1/12 * ([1/2 * (2A), [B, 1/2 * (2A)]]) + {0})) "  # (-1/12 [A, [A, B]] + 1/6 [B, [B, A]] - 1/12 [A, [A, B]])
            "* (2 * (1/2 * (2A)) + 2 * (B) + -1 * (B)))"  # * (2A + B)
            ), "(24) in my notes"

        for o in range(7):
            assert compute_extrapolation_error_term(o, s2, [1,2,3]) == 0, "extrapolation cancels 2l terms"
    finally:
        os.environ.clear()
        os.environ.update(env)
