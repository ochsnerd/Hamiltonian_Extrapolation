import os, tempfile

import sympy as sp

from .product_formula import S2
from .operator import strOperator


def test_S2():
    s2_1 = S2((strOperator('A'),), "s1")
    s2_2 = S2((strOperator('2A'), strOperator('B')), "s2")
    s2_3 = S2((strOperator('2A'), strOperator('2B'), strOperator('C')), "s3")

    assert s2_1.alphabet() == (1,)
    assert s2_2.alphabet() == (1, 2, 3)
    assert s2_3.alphabet() == (1, 2, 3, 4, 5)

    assert ({s: str(H_j) for s, H_j in s2_1.symbol_hamiltonian_map().items()}
            == {1: 'A'})
    assert ({s: str(H_j) for s, H_j in s2_2.symbol_hamiltonian_map().items()}
            == {1: '1/2 * (2A)', 2: 'B', 3: '1/2 * (2A)'})
    assert ({s: str(H_j) for s, H_j in s2_3.symbol_hamiltonian_map().items()}
            == {1: '1/2 * (2A)', 2: '1/2 * (2B)', 3: 'C', 4: '1/2 * (2B)', 5: '1/2 * (2A)'})

    assert s2_1.integration_symbols() == (1, 1)
    assert s2_2.integration_symbols() == (1, 3)
    assert s2_3.integration_symbols() == (1, 5)

    assert s2_2.ps_symbols(4) == (1, 3)
    assert s2_3.ps_symbols(99) == tuple(range(1, 100, 2))

    assert str(s2_1) == "S2_s1_J=1"
    assert str(s2_2) == "S2_s2_J=2"
    assert str(s2_3) == "S2_s3_J=3"


    # set $DATA_DIR to temp directory to force recomputation
    temp = tempfile.TemporaryDirectory()
    env = dict(os.environ)
    try:
        os.environ['DATA_DIR'] = temp.name
        assert str(s2_1.integration_constant(0)) == "{0}"
        assert str(s2_1.integration_constant(1)) == "-1 * (A)"
        assert str(s2_1.integration_constant(2)) == "{0}"

        assert str(s2_2.integration_constant(0)) == "{0}"
        assert str(s2_2.integration_constant(1)) == "2 * (B) + -1 * (B)"
        assert str(s2_2.integration_constant(2)) == "{0}"

        assert str(s2_3.integration_constant(0)) == "{0}"
        assert (str(s2_3.integration_constant(1)) ==
                "2 * (1/2 * (2B)) + 2 * (C) + -1 * (C)"), (
                    "Eq. (3.9) in Kobayashi1998")
        assert str(s2_3.integration_constant(2)) == "{0}"
        assert (str(s2_3.integration_constant(3)) ==
                "1/12 * ([1/2 * (2B), [C, 1/2 * (2B)]]) + 1/6 * ([C, [C, 1/2 * (2B)]]) + 1/12 * ([1/2 * (2B), [C, 1/2 * (2B)]]) + {0}"), (
            "Eq. (3.9) in Kobayashi1998")
    finally:
        os.environ.clear()
        os.environ.update(env)
