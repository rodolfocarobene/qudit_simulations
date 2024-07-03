import numpy as np
from cirq import Gate
from sympy import Matrix, Transpose
from sympy.physics.quantum import TensorProduct

# Pauli matrices (2x2)
sx = Matrix([[0, 1], [1, 0]])
sy = Matrix([[0, -1j], [1j, 0]])
sz = Matrix([[1, 0], [0, -1]])
si = Matrix([[1, 0], [0, 1]])  # identity 2x2

# Gamma matrices (4x4)
sy_gamma_1 = TensorProduct(sx, si)
sy_gamma_2 = TensorProduct(sy, si)
sy_gamma_3 = TensorProduct(sz, sx)
sy_gamma_4 = TensorProduct(sz, sy)
sy_gamma_5 = TensorProduct(sz, sz)
sy_id = TensorProduct(si, si)  # identity 4x4

# bra-kets for ququarts
ket_0 = Matrix([[1], [0], [0], [0]])  # empty state
bra_0 = Transpose(ket_0)
ket_1 = Matrix([[0], [1], [0], [0]])  # down state
bra_1 = Transpose(ket_1)
ket_2 = Matrix([[0], [0], [1], [0]])  # up state
bra_2 = Transpose(ket_2)
ket_3 = Matrix([[0], [0], [0], [1]])  # mixed state
bra_3 = Transpose(ket_3)

# two-qudit gate definition (more or less CNOT extended)
X = ket_0 * bra_1 + ket_1 * bra_2 + ket_2 * bra_3 + ket_3 * bra_0
sy_ucsum = (
    TensorProduct(Matrix(ket_0 * bra_0), X)
    + TensorProduct(Matrix(ket_1 * bra_1), X * X)
    + TensorProduct(Matrix(ket_2 * bra_2), X * X * X)
    + TensorProduct(Matrix(ket_3 * bra_3), X * X * X * X)
)
sy_ad_ucsum = sy_ucsum.adjoint()

# Cirq gate


# rotation around X for the ij couple
class X_P_ij(Gate):
    """Rotation around X for the ij couple states."""

    def __init__(self, i, j, phase=np.pi, *args, **kwargs):
        """Init and save attributes."""
        super().__init__(*args, **kwargs)
        self.i = i
        self.j = j
        self.phase = phase

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex_
        )
        matrix[self.i][self.i] = matrix[self.j][self.j] = np.cos(self.phase / 2)
        matrix[self.j][self.i] = matrix[self.i][self.j] = -1j * np.sin(self.phase / 2)
        return matrix

    def _circuit_diagram_info_(self, args):
        return f"X({self.phase:.2f})_{self.i}{self.j}"


class Y_P_ij(Gate):
    """Rotation around Y for the ij couple states."""

    def __init__(self, i, j, phase, *args, **kwargs):
        """Init and save attributes."""
        self.i = i
        self.j = j
        self.phase = phase
        super().__init__(*args, **kwargs)

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex_
        )
        matrix[self.i][self.i] = matrix[self.j][self.j] = np.cos(self.phase / 2)
        matrix[self.j][self.i] = np.sin(self.phase / 2)
        matrix[self.i][self.j] = -np.sin(self.phase / 2)
        return matrix

    def _circuit_diagram_info_(self, args):
        return f"Y({self.phase:.2f})_{self.i}{self.j}"


class Z_P_ij(Gate):
    """Rotation around Z for the ij couple states."""

    def __init__(self, i, j, phase, *args, **kwargs):
        """Init and save attributes."""
        self.i = i
        self.j = j
        self.phase = phase
        super().__init__(*args, **kwargs)

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex_
        )

        matrix[self.i][self.i] = np.exp(-1j * self.phase / 2)
        matrix[self.j][self.j] = np.exp(1j * self.phase / 2)
        return matrix

    def _circuit_diagram_info_(self, args):
        return f"Z({self.phase:.2f})_{self.i}{self.j}"


class Gamma1(Gate):
    """Gamma1 gate."""

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(sy_gamma_1)

    def _circuit_diagram_info_(self, args):
        return "Γ1"


class Gamma2(Gate):
    """Gamma2 gate."""

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(sy_gamma_2)

    def _circuit_diagram_info_(self, args):
        return "Γ2"


class Gamma3(Gate):
    """Gamma3 gate."""

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(sy_gamma_3)

    def _circuit_diagram_info_(self, args):
        return "Γ3"


class Gamma4(Gate):
    """Gamma4 gate."""

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(sy_gamma_4)

    def _circuit_diagram_info_(self, args):
        return "Γ4"


class Gamma5(Gate):
    """Gamma5 gate."""

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(sy_gamma_5)

    def _circuit_diagram_info_(self, args):
        return "Γ5"


class Id(Gate):
    """Identity4x4 gate."""

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(sy_id)

    def _circuit_diagram_info_(self, args):
        return "I"


class UCSUM(Gate):
    """UCSUM gate."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(sy_ucsum)

    def _circuit_diagram_info_(self, args):
        return ["c", "U+"]


class UCSUMDag(Gate):
    """UCSUM_adgjoint gate."""

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(sy_ad_ucsum)

    def _circuit_diagram_info_(self, args):
        return ["U+", "c"]
