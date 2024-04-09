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
from primitives import *

# %% [markdown]
# ## Helpers


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
                    final_str += f" {row * self.columns + (col+1):2d} "
                else:
                    final_str += f" {(row+1) * self.columns - (col):2d} "
            final_str += "\n"
        return final_str

    def __repr__(self):
        return self.__str__()

    def print(self):
        print(self.__str__())

    def step_int(self, qubit, t, v):
        # print(f"Adding on-site gate for {qubit}")
        return [
            Z_P_ij(0, 1, 2 * v * t)(qubit),
            Z_P_ij(0, 2, 2 * v * t)(qubit),
            Z_P_ij(0, 3, 2 * v * t)(qubit),
        ]

    def step_hop(self, qubit0, qubit1, t, J):
        # print(f"Adding int gate for {qubit0} - {qubit1}")
        qubits = [qubit0, qubit1]

        tau = t * J
        return [
            # U_1
            UCSUM()(qubits[0], qubits[1]),
            X_P_ij(0, 2, tau)(qubits[0]),
            X_P_ij(1, 3, -tau)(qubits[0]),
            # UCSUMDag()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, -np.pi)(qubits[0]),
            UCSUM()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, np.pi)(qubits[0]),
            # U_2
            Z_P_ij(0, 2, np.pi / 2)(qubits[0]),
            Z_P_ij(1, 3, -np.pi / 2)(qubits[0]),
            Z_P_ij(0, 2, np.pi / 2)(qubits[1]),
            Z_P_ij(1, 3, np.pi / 2)(qubits[1]),
            UCSUM()(qubits[0], qubits[1]),
            X_P_ij(0, 2, tau)(qubits[0]),
            X_P_ij(1, 3, tau)(qubits[0]),
            # UCSUMDag()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, -np.pi)(qubits[0]),
            UCSUM()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, np.pi)(qubits[0]),
            Z_P_ij(1, 3, np.pi / 2)(qubits[0]),
            Z_P_ij(0, 2, -np.pi / 2)(qubits[0]),
            Z_P_ij(1, 3, -np.pi / 2)(qubits[1]),
            Z_P_ij(0, 2, -np.pi / 2)(qubits[1]),
            # U_3
            Y_P_ij(0, 1, np.pi / 2)(qubits[0]),
            Y_P_ij(2, 3, -np.pi / 2)(qubits[0]),
            X_P_ij(0, 2, np.pi / 2)(qubits[0]),
            X_P_ij(1, 3, -np.pi / 2)(qubits[0]),
            Y_P_ij(0, 1, -np.pi / 2)(qubits[1]),
            Y_P_ij(2, 3, -np.pi / 2)(qubits[1]),
            Y_P_ij(0, 2, -np.pi / 2)(qubits[1]),
            Y_P_ij(1, 3, np.pi / 2)(qubits[1]),
            UCSUM()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, -tau)(qubits[0]),
            Y_P_ij(1, 3, -tau)(qubits[0]),
            # UCSUMDag()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, -np.pi)(qubits[0]),
            UCSUM()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, np.pi)(qubits[0]),
            X_P_ij(1, 3, np.pi / 2)(qubits[0]),
            X_P_ij(0, 2, -np.pi / 2)(qubits[0]),
            Y_P_ij(2, 3, np.pi / 2)(qubits[0]),
            Y_P_ij(0, 1, -np.pi / 2)(qubits[0]),
            Y_P_ij(1, 3, -np.pi / 2)(qubits[1]),
            Y_P_ij(0, 2, np.pi / 2)(qubits[1]),
            Y_P_ij(2, 3, np.pi / 2)(qubits[1]),
            Y_P_ij(0, 1, np.pi / 2)(qubits[1]),
            # U_4
            X_P_ij(0, 1, -np.pi / 2)(qubits[0]),
            X_P_ij(2, 3, np.pi / 2)(qubits[0]),
            Y_P_ij(0, 2, np.pi / 2)(qubits[0]),
            Y_P_ij(1, 3, -np.pi / 2)(qubits[0]),
            X_P_ij(0, 1, np.pi / 2)(qubits[1]),
            X_P_ij(2, 3, np.pi / 2)(qubits[1]),
            Y_P_ij(0, 2, np.pi / 2)(qubits[1]),
            Y_P_ij(1, 3, -np.pi / 2)(qubits[1]),
            UCSUM()(qubits[0], qubits[1]),
            X_P_ij(0, 2, -tau)(qubits[0]),
            X_P_ij(1, 3, -tau)(qubits[0]),
            # UCSUMDag()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, -np.pi)(qubits[0]),
            UCSUM()(qubits[0], qubits[1]),
            Y_P_ij(0, 2, np.pi)(qubits[0]),
            Y_P_ij(1, 3, np.pi / 2)(qubits[0]),
            Y_P_ij(0, 2, -np.pi / 2)(qubits[0]),
            X_P_ij(2, 3, -np.pi / 2)(qubits[0]),
            X_P_ij(0, 1, np.pi / 2)(qubits[0]),
            Y_P_ij(1, 3, np.pi / 2)(qubits[1]),
            Y_P_ij(0, 2, -np.pi / 2)(qubits[1]),
            X_P_ij(2, 3, -np.pi / 2)(qubits[1]),
            X_P_ij(0, 1, -np.pi / 2)(qubits[1]),
        ]

    def evolve(self, initial, temps, steps_for_step=10, J=-1, v=0):
        v = v / 4
        # warning

        self.tot_results = []
        self.t = temps

        for it, t in enumerate(temps):
            print(f"Computing t = {t:.2f} with {steps_for_step*it} steps")

            evolution_circuit = []

            if t != 0:
                tau = t / (steps_for_step * it)

                for _ in range(steps_for_step * it):
                    for qubit in self.qubits:
                        evolution_circuit.append(self.step_int(qubit, t=tau, v=v))

                    for idx in np.arange(0, self.rows * self.columns - 1, 2):
                        evolution_circuit.append(
                            self.step_hop(
                                self.qubits[idx],
                                self.qubits[idx + 1],
                                t=tau,
                                J=J,
                            )
                        )
                    for idx in np.arange(1, self.rows * self.columns - 1, 2):
                        evolution_circuit.append(
                            self.step_hop(
                                self.qubits[idx],
                                self.qubits[idx + 1],
                                t=tau,
                                J=J,
                            )
                        )
            measures = []
            for idx, qubit in enumerate(self.qubits):
                measures.append(measure(qubit, key=f"q{idx}"))

            circuit = Circuit([*initial, *evolution_circuit, *measures])
            print(f"Len: {len(circuit)}")

            result = QuditResult(sample(circuit, repetitions=100))
            self.tot_results.append((t, result))

    def plot(self, sites=None):

        if sites is None:
            sites = self.tot_results[0][1].sites

        for site in sites:
            tot_site = [res[1].sites[site] for res in self.tot_results]
            plt.plot(self.t, tot_site, "o-", label=f"N({site}) up")

        plt.legend()


# %%
qfh = QuditFermiHubbard(1, 4)
print("Studied lattice: \n", qfh)

J = -1
v = 0

t = np.arange(0, 3, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.qubits[0]),
    X_P_ij(0, 2)(qfh.qubits[1]),
    X_P_ij(0, 3)(qfh.qubits[2]),
    # X_P_ij(0, 2)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v)

qfh.plot()


# %%
