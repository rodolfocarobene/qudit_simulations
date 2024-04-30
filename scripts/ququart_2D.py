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
from scipy.linalg import expm

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
        return expm(
            np.array(
                -1j
                * self.t
                * (
                    4 * self.v * np.eye(4)
                    - 1j * sy_gamma_1 * sy_gamma_2
                    - 1j * sy_gamma_3 * sy_gamma_4
                    + sy_gamma_5
                ),
                dtype=np.complex128,
            )
        )

    def _circuit_diagram_info_(self, args):
        return "Hi"


class Nearhop(Gate):
    def __init__(self, J=1, t=0.1):
        self.J = J
        self.t = t
        super()

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return expm(
            np.array(
                -1j
                * self.t
                * (
                    self.J
                    / (2j)
                    * (
                        TensorProduct(sy_gamma_2 * sy_gamma_5, sy_gamma_1)
                        - TensorProduct(sy_gamma_1 * sy_gamma_5, sy_gamma_2)
                        + TensorProduct(sy_gamma_4 * sy_gamma_5, sy_gamma_3)
                        - TensorProduct(sy_gamma_3 * sy_gamma_5, sy_gamma_4)
                    )
                ),
                dtype=np.complex128,
            )
        )

    def _circuit_diagram_info_(self, args):
        return ["Hop", "Hop"]


class NotNearHop(Gate):
    def __init__(self, J=1, t=0.1, number_of_gammas=2):
        self.J = J
        self.t = t
        self.return_shape = tuple([4] * (2 + number_of_gammas))
        self.unitary_shape = tuple([16] * (2 + number_of_gammas))
        self.number_of_gammas = number_of_gammas
        super()

    def _qid_shape_(self):
        return self.return_shape

    def _unitary_(self):

        return_gammas = [sy_gamma_5] * self.number_of_gammas
        a = expm(
            np.array(
                -1j
                * self.t
                * (
                    self.J
                    / (2j)
                    * (
                        TensorProduct(
                            sy_gamma_2 * sy_gamma_5, sy_gamma_1, *return_gammas
                        )
                        - TensorProduct(
                            sy_gamma_1 * sy_gamma_5, sy_gamma_2, *return_gammas
                        )
                        + TensorProduct(
                            sy_gamma_4 * sy_gamma_5, sy_gamma_3, *return_gammas
                        )
                        - TensorProduct(
                            sy_gamma_3 * sy_gamma_5, sy_gamma_4, *return_gammas
                        )
                    )
                ),
                dtype=np.complex128,
            )
        )
        return a.reshape(self.unitary_shape)

    def _circuit_diagram_info_(self, args):
        return ["Hop", "Hop"] + ["G5"] * self.number_of_gammas


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

    def step_hop(self, t, J, qubit0, qubit1):
        # print(f"Adding int gate for {qubit0} - {qubit1}")
        return Nearhop(J, t)(qubit0, qubit1)

    def step_hop_gammas(self, t, J, qubit0, qubit1, *args):
        return NotNearHop(J, t, number_of_gammas=len(args))(qubit0, qubit1, *args)

    def evolve(self, initial, temps, steps_for_step=10, J=-1, v=0, repetitions=1000):
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
                                tau,
                                J,
                                self.qubits[idx],
                                self.qubits[idx + 1],
                            )
                        )
                    # vertical hopping terms
                    for idx in range(self.rows - 1):
                        for jdx in range(self.columns):
                            if not (
                                (jdx == self.columns - 1 and idx % 2 == 0)
                                or (jdx == 0 and idx % 2 == 1)
                            ):

                                if idx % 2 == 0:
                                    first_index = self.columns * idx + jdx
                                else:
                                    first_index = (self.columns) * (idx + 1) - (jdx + 1)
                                second_index = (
                                    (idx + 1) * self.columns * 2 - 1 - first_index
                                )

                                intermediate = [
                                    self.qubits[a]
                                    for a in range(first_index + 1, second_index)
                                ]
                                evolution_circuit.append(
                                    self.step_hop_gammas(
                                        tau,
                                        J,
                                        self.qubits[first_index],
                                        self.qubits[second_index],
                                        *intermediate,
                                    )
                                )

            measures = []
            for idx, qubit in enumerate(self.qubits):
                measures.append(measure(qubit, key=f"q{idx}"))

            circuit = Circuit([*initial, *evolution_circuit, *measures])
            log.info(f"Len: {len(circuit)}")

            result = QuditResult(sample(circuit, repetitions=repetitions))
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
repetitions = 1000

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    # X_P_ij(0, 1)(qfh.qubits[1]),
    # X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v, repetitions=repetitions)

qfh.plot()

# %% [markdown]
# ### Example: 3 up fermions

# %%
qfh = QuditFermiHubbard(2, 2)

J = -1
v = 0
repetitions = repetitions

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    X_P_ij(0, 1)(qfh.qubits[2]),
    X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v, repetitions=repetitions)

qfh.plot()

# %% [markdown]
# ### Example: 2 near up fermions

# %%
qfh = QuditFermiHubbard(2, 2)

J = -1
v = 0
repetitions = repetitions

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    X_P_ij(0, 1)(qfh.qubits[1]),
    # X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v, repetitions=repetitions)

qfh.plot()

# %% [markdown]
# ### Example: 2 not-near up fermions

# %%
qfh = QuditFermiHubbard(2, 2)

J = -1
v = 0
repetitions = repetitions

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    X_P_ij(0, 1)(qfh.qubits[2]),
    # X_P_ij(0, 1)(qfh.qubits[3]),
    # X_P_ij(0, 2)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v, repetitions=repetitions)

qfh.plot()
