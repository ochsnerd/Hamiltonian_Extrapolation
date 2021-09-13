import unittest.mock

from .operator import strOperator
from .product_formula import S1
from .bch_terms_casas import CasasMuruaReader


def test_CasasMuruaReader():
    file_content = """# downloaded from http://www.ehu.eus/ccwmuura/bch.html
    # Fernando Casas and Ander Murua. An efficient algorithm for computing the Baker-Campbell- Hausdorff series and some of its applications. Journal of Mathematical Physics, 50(3):033513, March 2009. arXiv: 0810.2656.
    # Number of elements per order 1, 2, 3, ..., 20
    2 1 2 3
    1        1       0       1              1
    2        2       0       1              1
    3        1       2       -1             2
    4        3       2       1              12
    5        1       3       -1             12
    6        4       2       0              1
    7        1       4       1              24
    8        1       5       0              1
    """
    mock_open = unittest.mock.mock_open(read_data=file_content)

    pf = S1([strOperator("A"), strOperator("B")], "")

    expected_output = [
        "1 * (A) + 0 + 1 * (B)",
        "-1/2 * ([A, B]) + 0",
        "1/12 * ([[A, B], B]) + 0 + -1/12 * ([A, [A, B]])",
        "0 * ([[[A, B], B], B]) + 0 + 1/24 * ([A, [[A, B], B]]) + 0 * ([A, [A, [A, B]]])"
    ]

    with unittest.mock.patch('builtins.open', mock_open):
        with CasasMuruaReader("", pf) as reader:
            for i, num_terms in zip(range(1, 5), (2, 1, 2, 3)):
                assert reader.current_order() == i
                assert len(list(reader.terms_of_current_order())) == num_terms

        with CasasMuruaReader("", pf) as reader:
            for term in expected_output:
                assert str(sum(reader.terms_of_current_order())), term
