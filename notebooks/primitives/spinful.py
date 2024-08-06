from primitives.primitives import *
from primitives.tv_model import *


class AddWeird(Gate):
    """Add two couple of fermions (r)+(r+x) - (r)+(r+x+1) in superposition."""

    def __init__(self, A=1, B=1, *args, **kwargs):
        self.A = A
        self.B = B
        super().__init__(*args, **kwargs)

    def _qid_shape_(self):
        return (4, 4, 4, 4, 4, 4)

    def _unitary_(self):
        B = TensorProduct(
            sy_id,
            Matrix(AddVerticalPair()._unitary_()),
            Matrix(AddDiagonalPair()._unitary_()),
        )
        A = TensorProduct(AddDiagonalPair()._unitary_(), AddDiagonalPair()._unitary_())
        return np.array(self.A * A + self.B * B, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["f+", "f+", "f+", "f+", "f+", "f+"]


class SpinfulNumberGate(Gate):
    """Double number gate."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        num = TensorProduct(number_matrix, number_matrix)
        return np.array(num, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["n(r)", "n(r')"]


def compute_state_exp_hamiltonian_fh(qudits, state, J, U):
    ham_k = 0
    ham_p = 0

    x_hop_couples = []
    y_hop_couples = []

    rows, cols, _ = qudits.shape
    for row in range(rows):
        for col in range(cols):
            if row != rows - 1:
                y_hop_couples.append([qudits[row][col][0], qudits[row + 1][col][0]])
                y_hop_couples.append([qudits[row][col][1], qudits[row + 1][col][1]])
            if col != cols - 1:
                x_hop_couples.append([qudits[row][col][0], qudits[row][col + 1][0]])
                x_hop_couples.append([qudits[row][col][1], qudits[row][col + 1][1]])

    # number term
    for row in range(rows):
        for col in range(cols):
            operator = U * SpinfulNumberGate()._unitary_()
            ham_p += expectation_value(
                state[1], [qudits[row][col][0], qudits[row][col][1]], operator
            )

    # x_hop term
    for couple in x_hop_couples:
        operator = J * x_hop_matrix
        ham_k += expectation_value(state[1], couple, operator)

    # y_hop term
    for couple in y_hop_couples:
        operator = J * y_hop_matrix
        ham_k += expectation_value(state[1], couple, operator)

    return ham_k, ham_p


class AddDiagTriple(Gate):
    """Add two couple of fermions (r)+(r+x) - (r)+(r+x+y) in superposition."""

    def __init__(self, A=1, B=1, *args, **kwargs):
        self.A = A
        self.B = B
        super().__init__(*args, **kwargs)

    def _qid_shape_(self):
        return (4, 4, 4)

    def _unitary_(self):
        A = TensorProduct(AddHorizontalPair()._unitary_(), sy_id)
        B = TensorProduct(HopXGate()._unitary_(), sy_id)
        B = B * TensorProduct(sy_id, Matrix(AddVerticalPair()._unitary_()))
        return np.array(self.A * A + self.B * B, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["f+(A+B)", "f+(A)", "f+(B)"]
