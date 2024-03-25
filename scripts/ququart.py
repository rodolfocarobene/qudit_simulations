# ## Ququart Simulation using Cirq

import matplotlib.pyplot as plt
import numpy as np
from cirq import Circuit, Gate, LineQid, measure, sample
from sympy import I, Matrix, Transpose, exp, eye
from sympy.physics.quantum import TensorProduct

# ### Sympy definitions

# +
sx = Matrix([[0, 1], [1, 0]])
sy = Matrix([[0, -I], [I, 0]])
sz = Matrix([[1, 0], [0, -1]])

si = Matrix([[1, 0], [0, 1]])

# +
gamma_1 = TensorProduct(sx, si)
gamma_2 = TensorProduct(sy, si)
gamma_3 = TensorProduct(sz, sx)
gamma_4 = TensorProduct(sz, sy)

gamma_5 = TensorProduct(sz, sz)

id = TensorProduct(si, si)

# +
ket_0 = Matrix([[1], [0], [0], [0]])
bra_0 = Transpose(ket_0)

ket_1 = Matrix([[0], [1], [0], [0]])
bra_1 = Transpose(ket_1)

ket_2 = Matrix([[0], [0], [1], [0]])
bra_2 = Transpose(ket_2)

ket_3 = Matrix([[0], [0], [0], [1]])
bra_3 = Transpose(ket_3)

X = ket_0 * bra_1 + ket_1 * bra_2 + ket_2 * bra_3 + ket_3 * bra_0

ucsum_sympy = (
    TensorProduct(Matrix(ket_0 * bra_0), id)
    + TensorProduct(Matrix(ket_1 * bra_1), X)
    + TensorProduct(Matrix(ket_2 * bra_2), X * X)
    + TensorProduct(Matrix(ket_3 * bra_3), X * X * X)
)
ad_ucsum_sympy = ucsum_sympy.adjoint()


# -

# ### Definition of the primitives


# +
class To0(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return "|0>"


class To1(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return "|1>"


class To2(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return "|2>"


class To3(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

    def _circuit_diagram_info_(self, args):
        return "|3>"


# -


class Gamma1(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(gamma_1)

    def _circuit_diagram_info_(self, args):
        return "Γ1"


class Gamma2(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(gamma_2)

    def _circuit_diagram_info_(self, args):
        return "Γ2"


class Gamma3(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(gamma_3)

    def _circuit_diagram_info_(self, args):
        return "Γ3"


class Gamma4(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(gamma_4)

    def _circuit_diagram_info_(self, args):
        return "Γ4"


class Gamma5(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(gamma_5)

    def _circuit_diagram_info_(self, args):
        return "Γ5"


class Id(Gate):
    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(id)

    def _circuit_diagram_info_(self, args):
        return "I"


# +
class UCSUM(Gate):
    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(ucsum_sympy)

    def _circuit_diagram_info_(self, args):
        return ["c", "U+"]


class UCSUMDag(Gate):
    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(ad_ucsum_sympy)

    def _circuit_diagram_info_(self, args):
        return ["U+", "c"]


# -


class Hint(Gate):
    def __init__(self, v=1, t=0.1):
        self.v = v
        self.t = t
        super()

    def _qid_shape_(self):
        return (4,)

    def _unitary_(self):
        return np.array(
            exp(
                -1j
                * self.t
                * (
                    4 * eye(4)
                    - 1j * gamma_1 * gamma_2
                    - 1j * gamma_3 * gamma_4
                    + gamma_5
                )
            )
        )

    def _circuit_diagram_info_(self, args):
        return "Hi"


class Hhop(Gate):
    def __init__(self, J=1, t=0.1):
        self.J = J
        self.t = t
        super()

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return np.array(
            exp(
                -1j
                * self.t
                * (
                    self.J
                    / (2j)
                    * (
                        TensorProduct(Matrix(gamma_2 * gamma_5), gamma_1)
                        + TensorProduct(Matrix(gamma_1 * gamma_5), gamma_2)
                        + TensorProduct(Matrix(gamma_4 * gamma_5), gamma_3)
                        + TensorProduct(Matrix(gamma_3 * gamma_5), gamma_4)
                    )
                )
            )
        )

    def _circuit_diagram_info_(self, args):
        return ["Hop(m)", "Hop(m+1)"]


# ### Helpers


class Result:
    def __init__(self, res):
        self.probabilities = {}

        for key in res.measurements:
            prob_key = {
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.0,
            }
            array = res.measurements[key].flatten()
            unique, counts = np.unique(array, return_counts=True)
            for idx, el in enumerate(unique):
                prob_key[el] = counts[idx] / len(array)
            self.probabilities[key] = prob_key


def evolve(temps, steps_for_step=10, J=1, v=0):
    tot_results = []

    for it, t in enumerate(temps):
        print(f"Computing t = {t:.2f} with {steps_for_step*(it+1)} steps")
        circuit = Circuit(
            [
                To1()(qubits[0]),
                To3()(qubits[1]),
                step(
                    qubits[0], qubits[1], t=t, J=J, v=v, reps=steps_for_step * (it + 1)
                ),
                measure(qubits[0], key="q0"),
                measure(qubits[1], key="q1"),
            ]
        )

        result = Result(sample(circuit, repetitions=50))
        tot_results.append(result)
    return tot_results


def step(qubit0, qubit1, t, J, v, reps):
    def single(squbit0, squbit1, st, sJ, sv):
        return [
            Hhop(J=sJ, t=st)(squbit0, squbit1),
            Hint(t=st)(squbit0),
            Hint(t=st)(squbit1),
        ]

    t = t / reps

    a = []
    for el in range(reps):
        a += single(qubit0, qubit1, t, J, v)
    return a


# ### 1D simulation
#
# - Γ1 = x20 + x31
# - Γ2 = y20 + y31
# - Γ3 = x01 - x32
# - Γ4 = y01 + y32
# - Γ5 = z01 - z23

# +
N = 2  # number of sites

qubits = LineQid.range(N, dimension=4)
qubits

# +
circuit = Circuit(
    [
        To1()(qubits[0]),
        To3()(qubits[1]),
        step(qubits[0], qubits[1], 4 / 5, 5, 0, 2),
        measure(qubits[0], key="q0"),
        measure(qubits[1], key="q1"),
    ]
)
print(circuit)

result = sample(circuit, repetitions=50)

my_res = Result(result)

print("\n")
print(result)
# -

# ### Evolution

# +
J = 1
v = 1

t = np.arange(0, 2 * J, J / 2)
results = evolve(t, steps_for_step=10, J=J, v=v)

# +
tot_up0 = [
    (res.probabilities["q0"][1] + res.probabilities["q0"][3] / 2) for res in results
]
tot_up1 = [
    (res.probabilities["q1"][1] + res.probabilities["q0"][3] / 2) for res in results
]

tot_down0 = [
    (res.probabilities["q0"][2] + res.probabilities["q0"][3] / 2) for res in results
]
tot_down1 = [
    (res.probabilities["q1"][2] + res.probabilities["q0"][3] / 2) for res in results
]
1
plt.plot(t, tot_up0, "o-", label="N(0)up")
plt.plot(t, tot_down0, "o-", label="N(0)down")

plt.plot(t, tot_up1, "^--", label="N(1)up")
plt.plot(t, tot_down1, "^--", label="N(1)down")

plt.legend()
# -