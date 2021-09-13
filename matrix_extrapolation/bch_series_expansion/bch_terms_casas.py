"""
Tools for reading the coefficient lists generated in the paper and translating them to operators.
Fernando Casas and Ander Murua. An efficient algorithm for computing the Baker-Campbell- Hausdorff series and some of its applications. Journal of Mathematical Physics, 50(3):033513, March 2009. arXiv: 0810.2656.

Files downloaded from
http://www.ehu.eus/ccwmuura/bch.html.
"""

from __future__ import annotations
from typing import Iterable, Tuple

import sympy as sp

from .operator import Operator
from .product_formula import ProductFormula, S1, S2


class CasasMuruaReader:
    """One-pass reader to read terms in the BCH expansion.

    Reads a file with the format

    # Some ignored header lines
    2 1 2 3 6 9 18 30 56 99 186 335 630 1161 2182 4080 7710 14532 27594 52377
    1        1       0     1               1
    2        2       0     1               1
    3        2       1     -1              2

    where the first line not starting with '#' contains the number of elements
    for every order (starting at 1), and each subsequent line contains
    a running number as identification (i), two numbers referencing previous lines
    (j,k), and two number representing the numerator and denominator of a fraction
    (n,d). Any whitespace is treated as seperators, any symbols other than
    whitespace and valid input to int() will result in an exception.

    The term i is then given by (n / d) * [j, k], with 1 <-> X, 2 <-> Y.
    As an example, the third term in the BCH-expansion log(e^Xe^Y) is
    characterized by (i=3, j=2, k=1, n=-1, d=2), so it is -1/2 [Y, X].

    Data files and a more extensive description can be found at
    http://www.ehu.eus/ccwmuura/bch.html.

    The typical usage of this class is:
    >>> pf = ...  # ProductFormula containing the Hamiltonian terms
    >>> with CasasMuruaReader("/path/to/bchHall20.dat", pf) as reader:
    >>>     for i in range(1, 11):
    >>>         reader.current_order() == i
    >>>         K_i = sum(reader.terms_of_current_order())

    The file will stay opened during the livetime of this object. every line
    will be read exactly once, in order, and there is no way to get back to
    a previous position. This is done because the recursive nature of the
    file content requires in-order-reading to be efficient.
    """
    def __init__(self, filename: str, product_formula: ProductFormula) \
            -> None:
        assert product_formula.J == 2

        # store operators as first entries in E
        self.E = {1: product_formula.H_terms[0], 2: product_formula.H_terms[1]}

        # read header, store number of entries per order and length of header
        self.filename = filename
        self.datfile = open(filename, mode='r')
        for line in self.datfile:
            if line.lstrip().startswith('#'):
                continue
            self.num_elements = [int(n) for n in line.split()]
            break

        self.max_order = len(self.num_elements)
        self._curr_order = 1

    def __enter__(self) -> CasasMuruaReader:
        return self

    def __exit__(self, *args):
        self.datfile.close()

    def current_order(self) -> int:
        return self._curr_order

    def terms_of_current_order(self) -> Iterable[Operator]:
        """Return a generator yielding all terms of the current order.

        Reads as many lines as is indicated by the first line of the file,
        converts them to operators and returns them one by one.
        """
        if self._curr_order > self.max_order:
            raise ValueError(f"Only terms up to order {self.max_order} in the specified file.")

        for lineno, line in enumerate(self.datfile):
            i, j, k, n, d = self._parse_line(line)
            if self._curr_order != 1:
                # except for the first-order terms, we have to compute the commutator
                self.E[i] = self.E[j].commutator(self.E[k])
            yield sp.Rational(n, d) * self.E[i]

            if lineno + 1 >= self.num_elements[self._curr_order - 1]:
                # we've read all terms of the current order,
                # increment and stop iterating
                self._curr_order += 1
                return

    def _parse_line(self, line):
        return tuple(int(n) for n in line.split())


def compute_BCH_terms(product_formula: ProductFormula,
                      max_order: int) \
        -> Tuple[Operator]:
    """'Compute' BCH terms up to order max_order"""
    if not product_formula.ps_symbols(max_order):
        return ()

    # this is not very elegant
    import os
    datadir = os.path.join(os.path.dirname(os.path.abspath(__file__))
                           , '..', '..', 'data')
    files = {S1: os.path.join(datadir, 'bchHall20.dat'),
             S2: os.path.join(datadir, 'sbchHall19.dat')}

    terms = []
    with CasasMuruaReader(files[type(product_formula)], product_formula) as reader:
        for _ in range(product_formula.ps_symbols(max_order)[-1]):
            if reader.current_order() in product_formula.ps_symbols(max_order):
                terms += [sum(reader.terms_of_current_order())]
            else:
                # current order is not in ps_terms (and thus will be 0)
                # still need to iterate through to read all lines since they
                # will be used after
                for __ in reader.terms_of_current_order():
                    pass

    return tuple(terms)
