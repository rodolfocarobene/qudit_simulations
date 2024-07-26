from primitives.primitives import *
from scipy import sparse
from scipy.linalg import expm


def commute(a, b, rtol=1e-5):
    a = np.array(a, dtype=np.complex128)
    b = np.array(b, dtype=np.complex128)

    c = a @ b - b @ a
    return (np.isclose(c, np.zeros(c.shape), rtol)).all()


def expectation_value(result, qudits, operator):
    rho = result.density_matrix_of(qudits)

    srho = sparse.coo_array(np.array(rho, dtype=np.complex128))
    srho /= srho.trace()
    shop = sparse.coo_array(np.array(operator, dtype=np.complex128))
    val = srho @ shop

    tr = val.trace()
    if np.imag(tr) >= 1e-5:
        print(f"Casting {tr} to real")
    return np.real(tr)


def compute_state_exp_numbers(qudits, state):
    nums = []
    for qudit in qudits.flatten():
        val = expectation_value(state[1], [qudit], number_matrix)
        nums.append(val)
    return nums


def compute_state_exp_hamiltonian(qudits, state, T, V):
    ham_k = 0
    ham_p = 0

    x_hop_couples = []
    y_hop_couples = []

    rows, cols = qudits.shape
    for row in range(rows):
        for col in range(cols):
            if row != rows - 1:
                y_hop_couples.append([qudits[row][col], qudits[row + 1][col]])
            if col != cols - 1:
                x_hop_couples.append([qudits[row][col], qudits[row][col + 1]])

    # number term
    couples = x_hop_couples + y_hop_couples
    for couple in couples:
        operator = V * double_number_matrix
        ham_p += expectation_value(state[1], couple, operator)

    # x_hop term
    for couple in x_hop_couples:
        operator = T * x_hop_matrix
        ham_k += expectation_value(state[1], couple, operator)

    # y_hop term
    for couple in y_hop_couples:
        operator = T * y_hop_matrix
        ham_k += expectation_value(state[1], couple, operator)

    return ham_k, ham_p


def generate_plaquette_constraints(qudits):
    plaquettes = []
    rows, cols = qudits.shape
    for row in range(rows):
        for col in range(cols):
            if row != rows - 1 and col != cols - 1:
                pl = [
                    qudits[row][col],
                    qudits[row][col + 1],
                    qudits[row + 1][col + 1],
                    qudits[row + 1][col],
                ]
                plaquettes.append(pl)
    return plaquettes


x_hop_matrix = 1j * TensorProduct(
    sy_gamma_1 * sy_gamma_5, sy_gamma_2
) - 1j * TensorProduct(sy_gamma_1, sy_gamma_2 * sy_gamma_5)
x_hop_matrix /= 2


class HopXGate(Gate):
    """Horizontal hopping gate."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(x_hop_matrix, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["H_x(r)", "H_x(r+1)"]


y_hop_matrix = 1j * TensorProduct(
    sy_gamma_3 * sy_gamma_5, sy_gamma_4
) - 1j * TensorProduct(sy_gamma_3, sy_gamma_4 * sy_gamma_5)
y_hop_matrix /= 2


class HopYGate(Gate):
    """Vertical hopping gate."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(y_hop_matrix, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["H_y(r)", "H_y(r+1)"]


number_matrix = 0.5 * (sy_id - sy_gamma_5)


class NumberGate(Gate):
    """Number gate."""

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(number_matrix, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["n(r)"]


double_number_matrix = TensorProduct(number_matrix, number_matrix)


class DoubleNumberGate(Gate):
    """Double number gate."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(double_number_matrix, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["n(r)", "n(r+1)"]


def evolve_gate(gate):
    class evolved(Gate):
        def __init__(self, C=1, t=0.1, *args, **kwargs):
            gate_mat = np.array(gate()._unitary_(), dtype=np.complex128)
            if (gate_mat != np.conjugate(gate_mat.T)).any():
                print("Not hermitian")
            self.C = C  # coefficient
            self.t = t  # evolution time
            self.matrix = np.array(
                expm(-1j * self.t * self.C * gate_mat), dtype=np.complex128
            )
            super().__init__(*args, **kwargs)

        def _qid_shape_(self):
            return gate()._qid_shape_()

        def _circuit_diagram_info_(self, args):
            return gate()._circuit_diagram_info_(args)

        def _unitary_(self):
            return self.matrix

    return evolved


G = np.array(
    TensorProduct(
        sy_gamma_1 * sy_gamma_3,
        sy_gamma_2 * sy_gamma_3,
        sy_gamma_4 * sy_gamma_2,
        sy_gamma_1 * sy_gamma_4,
    ),
    dtype=np.complex128,
)


class Project_Constraint_minus(Gate):
    """Gate that prepare an initial state, from the coefficients."""

    def _qid_shape_(self):
        return (4, 4, 4, 4)

    def _unitary_(self):
        return (np.eye(4**4) - G) / 2

    def _circuit_diagram_info_(self, args):
        return ["P-", "P-", "P-", "P-"]


class Project_Constraint(Gate):
    """Gate that prepare an initial state, from the coefficients."""

    def _qid_shape_(self):
        return (4, 4, 4, 4)

    def _unitary_(self):
        return (np.eye(4**4) + G) / 2

    def _circuit_diagram_info_(self, args):
        return ["P+", "P+", "P+", "P+"]


class AddHorizontalPair(Gate):
    """Add pair of fermions (r)+(r+x)."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(
            1j
            * (1 / 4)
            * TensorProduct(sy_gamma_1, sy_gamma_2)
            * TensorProduct(sy_id + sy_gamma_5, sy_id + sy_gamma_5),
            dtype=np.complex128,
        )

    def _circuit_diagram_info_(self, args):
        return ["f+", "f+"]


class AddVerticalPair(Gate):
    """Add pair of fermions (r)+(r+y)."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        create = (
            1j
            * (1 / 4)
            * TensorProduct(sy_gamma_3, sy_gamma_4)
            * TensorProduct(sy_id + sy_gamma_5, sy_id + sy_gamma_5)
        )
        return np.array(create, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["f+", "f+"]


class AddDiagonalPair(Gate):
    """Add pair of fermions (r)+(r+x+y)."""

    def _qid_shape_(self):
        return (4, 4, 4)

    def _unitary_(self):
        create = (
            1j
            * (1 / 4)
            * TensorProduct(sy_gamma_1, sy_gamma_2)
            * TensorProduct(sy_id + sy_gamma_5, sy_id + sy_gamma_5)
        )
        # (r, r+x, r+x+y)
        # hop vertical
        create_hopped = TensorProduct(sy_id, y_hop_matrix) * TensorProduct(
            create, sy_id
        )
        return np.array(create_hopped, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["f+", "/", "f+"]


class AddAntiDiagonalPair(Gate):
    """Add pair of fermions (r+x)+(r+y)."""

    def _qid_shape_(self):
        return (4, 4, 4)

    def _unitary_(self):
        create = (
            1j
            * (1 / 4)
            * TensorProduct(sy_gamma_3, sy_gamma_4)
            * TensorProduct(sy_id + sy_gamma_5, sy_id + sy_gamma_5)
        )
        # (r, r+x, r+x+y, r+y)
        # hop vertical
        create_hopped = TensorProduct(sy_id, -x_hop_matrix) * TensorProduct(
            create, sy_id
        )
        return np.array(create_hopped, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["f+", "/", "f+"]


class AddCoupledPair(Gate):
    """Add two couple of fermions (r)+(r+x) - (r)+(r+x+1) in superposition."""

    def __init__(self, A=1, B=1, *args, **kwargs):
        self.A = A
        self.B = B
        super().__init__(*args, **kwargs)

    def _qid_shape_(self):
        return (4, 4, 4)

    def _unitary_(self):
        A = TensorProduct(AddHorizontalPair()._unitary_(), sy_id)
        B = TensorProduct(sy_id, Matrix(AddVerticalPair()._unitary_()))
        return np.array(self.A * A + self.B * B, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["f+(A)", "f+(A+B)", "f+(B)"]


class AddTriple(Gate):
    """Add two couple of fermions (r)+(r+x) - (r)+(r+x+1) in superposition."""

    def __init__(self, A=1, B=1, *args, **kwargs):
        self.A = A
        self.B = B
        super().__init__(*args, **kwargs)

    def _qid_shape_(self):
        return (4, 4, 4)

    def _unitary_(self):
        A = TensorProduct(AddHorizontalPair()._unitary_(), sy_id)
        B = TensorProduct(sy_id, Matrix(HopXGate()._unitary_())) * TensorProduct(
            AddHorizontalPair()._unitary_(), sy_id
        )
        return np.array(self.A * A + self.B * B, dtype=np.complex128)

    def _circuit_diagram_info_(self, args):
        return ["f+(A+B)", "f+(A)", "f+(B)"]
