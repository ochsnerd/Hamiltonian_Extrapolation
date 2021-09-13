import numpy as np

from qiskit.quantum_info import Pauli

from .operator import PauliOperator


def test_PauliOperator():
    XP = Pauli("X")
    YP = Pauli("Y")
    ZP = Pauli("Z")

    XM = XP.to_matrix()
    YM = YP.to_matrix()
    ZM = ZP.to_matrix()
    zeroM = XM - XM

    # constructor
    assert PauliOperator([], []).to_matrix() == 0
    assert PauliOperator.from_PauliOp(PauliOperator([], [])).to_matrix() == 0

    assert np.array_equal(PauliOperator([XP], [1]).to_matrix(), XM)
    assert np.array_equal(PauliOperator([XP], [2]).to_matrix(), 2 * XM)
    assert np.array_equal(PauliOperator([XP, YP, ZP], [1, 2, 3]).to_matrix(),
                          1*XM + 2*YM + 3*ZM)

    XO = PauliOperator([XP], [1])
    YO = PauliOperator([YP], [1])
    ZO = PauliOperator([ZP], [1])
    X2O = PauliOperator([XP], [2])
    XYZO = PauliOperator([XP, YP, ZP], [1, 1, 1])

    # addition
    assert np.array_equal((XO + X2O).to_matrix(), XM + XM + XM)
    assert np.array_equal((YO + ZO).to_matrix(), YM + ZM)
    assert np.array_equal(((XO + YO) + ZO).to_matrix(), XM + YM + ZM)
    assert np.array_equal(XYZO.to_matrix(), (XO + YO + ZO).to_matrix())

    # scalar mul
    assert np.array_equal((XO * 2).to_matrix(), 2 * XM)
    assert np.array_equal((2 * XO).to_matrix(), 2 * XM)
    assert np.array_equal((2 * X2O).to_matrix(), 4 * XM)

    # subtraction
    assert np.array_equal((XO - ZO).to_matrix(), XM - ZM)
    assert np.array_equal((XYZO - XO - YO - ZO).to_matrix(), zeroM)

    # operator mul
    assert np.array_equal((XO * XO).to_matrix(), XM @ XM)
    assert np.array_equal((YO * ZO).to_matrix(), YM @ ZM)
    assert np.array_equal(((XO * YO) * ZO).to_matrix(), XM @ YM @ ZM)
    assert np.array_equal((XYZO * X2O).to_matrix(), 2 * (XM + YM + ZM) @ XM)

    # commutator
    assert np.array_equal(XO.commutator(XO).to_matrix(), zeroM)
    assert np.array_equal(XO.commutator(YO).to_matrix(), 2j * ZM)

    XX = PauliOperator([Pauli("XX")], [1])
    XY = PauliOperator([Pauli("XY")], [1])
    commXXXY = np.kron(XM, XM) @ np.kron(XM, YM) - np.kron(XM, YM) @ np.kron(XM, XM)

    assert np.array_equal(XX.commutator(XY).to_matrix(), commXXXY)

    # reduce internal representation to just one Pauli
    assert len(PauliOperator([Pauli("XZI")], [1]).commutator(PauliOperator([Pauli("ZZY")], [1]).commutator(PauliOperator([Pauli("ZXY")], [1]))).s_P) == 1, "Reduce internal represenation to just one Pauli for 1-term commutators"

    # 3rd order error term of 1 trotter step of 2-site open Ising-chain
    H_Z = PauliOperator([Pauli("ZZ")], [1])
    H_X = PauliOperator([Pauli("IX")], [1]) + PauliOperator([Pauli("XI")], [1])
    assert np.allclose(((H_X.commutator(H_X.commutator(H_Z)) -
                         0.5 * H_Z.commutator(H_Z.commutator(H_X))) * (1/12)).to_matrix(),
                       np.array([[ 2/3, -1/6, -1/6,  2/3],
                                 [-1/6, -2/3, -2/3, -1/6],
                                 [-1/6, -2/3, -2/3, -1/6],
                                 [ 2/3, -1/6, -1/6,  2/3]], dtype=complex))
