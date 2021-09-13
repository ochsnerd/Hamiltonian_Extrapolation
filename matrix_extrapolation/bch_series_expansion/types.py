from typing import Tuple

from sympy import Rational

BCH_Symbol = int
BCH_Word = Tuple[BCH_Symbol, ...]
BCH_Subterm = Tuple[Rational, BCH_Word]

PS_Symbol = int
PS_Word = Tuple[PS_Symbol, ...]
PS_Subterm = Tuple[Rational, PS_Word]
