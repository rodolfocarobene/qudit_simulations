# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: qudit
#     language: python
#     name: qudit
# ---

# %% [markdown]
# ## Ququart Simulation using Cirq


# %%
import matplotlib.pyplot as plt
import numpy as np
from cirq import Circuit, LineQid, measure, sample
from logger import log
from primitives import *
from sympy import exp

# %% [markdown]
# ## Helpers


# %%
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
                    4 * self.v * np.eye(4)
                    - 1j * sy_gamma_1 * sy_gamma_2
                    - 1j * sy_gamma_3 * sy_gamma_4
                    + sy_gamma_5
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
                        TensorProduct(Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1)
                        - TensorProduct(Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2)
                        + TensorProduct(Matrix(sy_gamma_4 * sy_gamma_5), sy_gamma_3)
                        - TensorProduct(Matrix(sy_gamma_3 * sy_gamma_5), sy_gamma_4)
                    )
                )
            )
        )

    def _circuit_diagram_info_(self, args):
        return ["Hop(m)", "Hop(m+1)"]


class GammaHop(Gate):
    def __init__(self, J=1, t=0.1):
        self.J = J
        self.t = t
        super()

    def _qid_shape_(self):
        return (4, 4, 4, 4)

    def _unitary_(self):
        def myexp(a):
            shape = a.shape
            for indices in np.ndindex(shape):
                a[indices] = np.exp(complex(a[indices]))
            return a

        a = np.array(
            myexp(
                -1j
                * self.t
                * (
                    self.J
                    / (2j)
                    * (
                        TensorProduct(
                            TensorProduct(
                                TensorProduct(
                                    Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1
                                ),
                                sy_gamma_5,
                            ),
                            sy_gamma_5,
                        )
                        - TensorProduct(
                            TensorProduct(
                                TensorProduct(
                                    Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2
                                ),
                                sy_gamma_5,
                            ),
                            sy_gamma_5,
                        )
                        + TensorProduct(
                            TensorProduct(
                                TensorProduct(
                                    Matrix(sy_gamma_4 * sy_gamma_5), sy_gamma_3
                                ),
                                sy_gamma_5,
                            ),
                            sy_gamma_5,
                        )
                        - TensorProduct(
                            TensorProduct(
                                TensorProduct(
                                    Matrix(sy_gamma_3 * sy_gamma_5), sy_gamma_4
                                ),
                                sy_gamma_5,
                            ),
                            sy_gamma_5,
                        )
                    )
                )
            )
        )
        return a.reshape(16, 16, 16, 16)

    def _circuit_diagram_info_(self, args):
        return ["H", "H", "G5", "G5"]


# %%
class QuditResult:
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

        self.sites = {}
        for site in self.probabilities:
            self.sites[site] = (
                self.probabilities[site][1] + self.probabilities[site][3] / 2
            )


# %%
class QuditFermiHubbard:
    def __init__(self, L=1, M=2):
        """Initialize qudit system, L x M."""
        self.rows = L
        self.columns = M

        self.qubits = LineQid.range(L * M, dimension=4)
        self.tot_results = []

    def __str__(self):
        final_str = ""
        for row in range(self.rows):
            for col in range(self.columns):
                if row % 2 == 0:
                    final_str += f"{row * self.columns + (col+1) - 1} "
                else:
                    final_str += f"{(row+1) * self.columns - (col) - 1} "
            final_str += "\n"
        return final_str

    def __repr__(self):
        return self.__str__()

    def print(self):
        print(self.__str__())

    def step_int(self, qubit, t, v):
        # print(f"Adding on-site gate for {qubit}")
        return Hint(v, t)(qubit)

    def step_hop(self, qubit0, qubit1, t, J):
        # print(f"Adding int gate for {qubit0} - {qubit1}")
        return Hhop(J, t)(qubit0, qubit1)

    def step_hop_gammas(self, qubit0, qubit1, qubit2, qubit3, t, J):
        # print(f"Adding int gate for {qubit0} - {qubit1}")
        return Hhop_gammas(J, t)(qubit0, qubit1, qubit2, qubit3)

    def evolve(self, initial, temps, steps_for_step=10, J=-1, v=0):
        v = v / 4
        # warning

        self.tot_results = []
        self.t = temps

        for it, t in enumerate(temps):
            log.info(f"Computing t = {t:.2f} with {steps_for_step*it} steps")

            evolution_circuit = []

            if t != 0:
                tau = t / (steps_for_step * it)

                for _ in range(steps_for_step * it):

                    for qubit in self.qubits:
                        # on-site part
                        evolution_circuit.append(self.step_int(qubit, t=tau, v=v))

                    # horizontal hopping terms
                    for idx in np.arange(0, self.rows * self.columns - 1):
                        evolution_circuit.append(
                            self.step_hop(
                                self.qubits[idx],
                                self.qubits[idx + 1],
                                t=tau,
                                J=J,
                            )
                        )
                    # vertical hopping terms
                    evolution_circuit.append(
                        GammaHop(
                            t=tau,
                            J=J,
                        )(
                            self.qubits[0],
                            self.qubits[3],
                            self.qubits[1],
                            self.qubits[2],
                        )
                    )
                    """
                    evolution_circuit.append(
                        GammaHop(J, tau)(self.qubits[1])
                    )
                    evolution_circuit.append(
                        GammaHop(J, tau)(self.qubits[2])
                    )
                    evolution_circuit.append(
                        Hhop(J, tau)(self.qubits[0], self.qubits[3])
                    )
                    evolution_circuit.append(
                        ExpGamma(J, tau)(self.qubits[1])
                    )
                    evolution_circuit.append(
                        ExpGamma(J, tau)(self.qubits[2])
                    )
                    """

            measures = []
            for idx, qubit in enumerate(self.qubits):
                measures.append(measure(qubit, key=f"q{idx}"))

            circuit = Circuit([*initial, *evolution_circuit, *measures])
            log.info(f"Len: {len(circuit)}")

            result = QuditResult(sample(circuit, repetitions=10))
            self.tot_results.append((t, result))

    def plot(self, sites=None):

        if sites is None:
            sites = self.tot_results[0][1].sites

        for site in sites:
            tot_site = [res[1].sites[site] for res in self.tot_results]
            plt.plot(self.t, tot_site, "o-", label=f"N({site}) up")

        plt.legend()


# %%
qfh = QuditFermiHubbard(2, 2)
print(f"Studied lattice: \n{qfh}")

# %% [markdown]
# ### Example: single up fermion

# %%
J = -1
v = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    # X_P_ij(0, 1)(qfh.qubits[1]),
    # X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v)

qfh.plot()

# %% [markdown]
# ### Example: 3 up fermions

# %%
qfh = QuditFermiHubbard(2, 2)

J = -1
v = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    X_P_ij(0, 1)(qfh.qubits[2]),
    X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v)

qfh.plot()

# %% [markdown]
# ### Example: 2 near up fermions

# %%
qfh = QuditFermiHubbard(2, 2)

J = -1
v = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    X_P_ij(0, 1)(qfh.qubits[1]),
    # X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v)

qfh.plot()

# %% [markdown]
# ### Example: 2 not-near up fermions

# %%
qfh = QuditFermiHubbard(2, 2)

J = -1
v = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    X_P_ij(0, 1)(qfh.qubits[2]),
    # X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v)

qfh.plot()

# %%
c

# %%
a = myexp(
    -1j
    * t
    * (
        J
        / 2j
        * (
            np.tensordot(sy_gamma_2 * sy_gamma_5, sy_gamma_1, axes=0)
            - np.tensordot(sy_gamma_1 * sy_gamma_5, sy_gamma_2, axes=0)
            + np.tensordot(sy_gamma_4 * sy_gamma_5, sy_gamma_3, axes=0)
            - np.tensordot(sy_gamma_3 * sy_gamma_5, sy_gamma_4, axes=0)
        )
    )
)

b = myexp(-1j * t * J / 2j * sy_gamma_5)
c = np.tensordot(a, np.tensordot(b, b, axes=0), axes=0)

# %%
t = 1
J = -1

a = np.array(
    myexp(
        -1j
        * t
        * (
            J
            / (2j)
            * (
                TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
                - TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
                + TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_4 * sy_gamma_5), sy_gamma_3),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
                - TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_3 * sy_gamma_5), sy_gamma_4),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
            )
        )
    )
)

# %%
b = GammaHop()._unitary_().reshape(16, 16, 16, 16)

# %%
t = 0.1
J = -1

a = np.array(
    myexp(
        -1j
        * t
        * (
            J
            / (2j)
            * (
                TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
                - TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
                + TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_4 * sy_gamma_5), sy_gamma_3),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
                - TensorProduct(
                    TensorProduct(
                        TensorProduct(Matrix(sy_gamma_3 * sy_gamma_5), sy_gamma_4),
                        sy_gamma_5,
                    ),
                    sy_gamma_5,
                )
            )
        )
    )
)

# %%
a == b.reshape(256, 256)

# %%
a

# %%
b

# %%
k = (
    -1j
    * t
    * (
        J
        / (2j)
        * (
            TensorProduct(
                TensorProduct(
                    TensorProduct(Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1),
                    sy_gamma_5,
                ),
                sy_gamma_5,
            )
            - TensorProduct(
                TensorProduct(
                    TensorProduct(Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2),
                    sy_gamma_5,
                ),
                sy_gamma_5,
            )
            + TensorProduct(
                TensorProduct(
                    TensorProduct(Matrix(sy_gamma_4 * sy_gamma_5), sy_gamma_3),
                    sy_gamma_5,
                ),
                sy_gamma_5,
            )
            - TensorProduct(
                TensorProduct(
                    TensorProduct(Matrix(sy_gamma_3 * sy_gamma_5), sy_gamma_4),
                    sy_gamma_5,
                ),
                sy_gamma_5,
            )
        )
    )
)


# %%
k = np.array(k)

# %%
expm(k.astype(np.complex128))

# %%
from scipy.linalg import expm

t = 1
J = -1


def mycexp(s):
    curr = 0
    for i in range(1, 10):
        L_curr = s
        for j in range(i - 1):
            L_curr = L_curr * s
        curr += L_curr / np.math.factorial(i)
    return curr


m = np.array(
    exp(
        -1j
        * t
        * (
            J
            / (2j)
            * (
                TensorProduct(Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1)
                - TensorProduct(Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2)
                + TensorProduct(Matrix(sy_gamma_4 * sy_gamma_5), sy_gamma_3)
                - TensorProduct(Matrix(sy_gamma_3 * sy_gamma_5), sy_gamma_4)
            )
        )
    )
)
n = np.array(
    mycexp(
        -1j
        * t
        * (
            J
            / (2j)
            * (
                TensorProduct(Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1)
                - TensorProduct(Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2)
                + TensorProduct(Matrix(sy_gamma_4 * sy_gamma_5), sy_gamma_3)
                - TensorProduct(Matrix(sy_gamma_3 * sy_gamma_5), sy_gamma_4)
            )
        )
    )
)
m == n

# %%
m

# %%
n

# %%
