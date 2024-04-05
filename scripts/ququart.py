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
from cirq import Circuit, Gate, LineQid, measure, sample
from primitives import *
from sympy import exp

# %% [markdown]
# ## Helpers


# %%
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

    def step_int(self, qubit, t, v, reps):
        if t == 0:
            return []

        def single(tau, sv):
            return [
                Z_P_ij(0, 1, 2 * sv * tau)(qubit),
                Z_P_ij(0, 2, 2 * sv * tau)(qubit),
                Z_P_ij(0, 3, 2 * sv * tau)(qubit),
            ]

        t = t / reps

        a = []
        for el in range(reps):
            a += single(t, v)
        return a

    def step_hop(self, qubit0, qubit1, t, J, reps):
        qubits = self.qubits
        if t == 0:
            return []

        def single(tau, sJ):
            tau = tau * sJ
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

        t = t / reps

        a = []
        for el in range(reps):
            a += single(t, J)
        return a

    def evolve(self, initial, temps, steps_for_step=10, J=-1, v=0):
        self.tot_results = []
        self.t = temps

        for it, t in enumerate(temps):
            print(f"Computing t = {t:.2f} with {steps_for_step*it} steps")

            evolution = []

            for idx in range(self.rows * self.columns):
                evolution.append(
                    self.step_int(self.qubits[idx], t=t, v=v, reps=steps_for_step * it)
                )

            for idx in np.arange(0, self.rows * self.columns, 2):
                evolution.append(
                    self.step_hop(
                        self.qubits[idx],
                        self.qubits[idx + 1],
                        t=t,
                        J=J,
                        reps=steps_for_step * it,
                    )
                )
            for idx in np.arange(1, self.rows * self.columns - 1, 2):
                evolution.append(
                    self.step_hop(
                        self.qubits[idx],
                        self.qubits[idx + 1],
                        t=t,
                        J=J,
                        reps=steps_for_step * it,
                    )
                )
            measures = []
            for idx in range(self.rows * self.columns):
                measures.append(measure(self.qubits[idx], key=f"q{idx}"))

            circuit = Circuit([*initial, *evolution, *measures])
            print(f"Len: {len(circuit)}")

            result = Result(sample(circuit, repetitions=100))
            self.tot_results.append(result)

    def plot(self):

        tot_up0 = [
            (res.probabilities["q0"][1] + res.probabilities["q0"][3] / 2)
            for res in self.tot_results
        ]
        tot_up1 = [
            (res.probabilities["q1"][1] + res.probabilities["q1"][3] / 2)
            for res in self.tot_results
        ]

        tot_down0 = [
            (res.probabilities["q0"][2] + res.probabilities["q0"][3] / 2)
            for res in self.tot_results
        ]
        tot_down1 = [
            (res.probabilities["q1"][2] + res.probabilities["q1"][3] / 2)
            for res in self.tot_results
        ]

        plt.plot(self.t, tot_up0, "o-", label="N(0)up")
        plt.plot(self.t, tot_down0, "o-", label="N(0)down")

        plt.plot(self.t, tot_up1, "^--", label="N(1)up")
        plt.plot(self.t, tot_down1, "^--", label="N(1)down")

        plt.legend()


# %%
qfh = QuditFermiHubbard()
print("Studied lattice: \n", qfh)

J = -1
v = -1

t = np.arange(0, 4, 1 / 2)

initial = [
    # set qudit 0 to |1> (up spin)
    X_P_ij(0, 1)(qfh.qubits[0]),
    # set qudit 1 to |3> (mixed state)
    X_P_ij(0, 3)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v)

qfh.plot()


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

    def step(self, qubit0, qubit1, t, J, v, reps):
        def single(squbit0, squbit1, st, sJ, sv):
            return [
                Hhop(J=sJ, t=st)(squbit0, squbit1),
                Hint(t=st)(squbit0),
                Hint(t=st)(squbit1),
            ]

        if reps == 0:
            return []
        t = t / reps

        a = []
        for el in range(reps):
            a += single(qubit0, qubit1, t, J, v)
        return a

    def evolve(self, initial, temps, steps_for_step=10, J=-1, v=0):
        qubits = self.qubits
        self.tot_results = []
        self.t = temps

        for it, t in enumerate(temps):
            print(f"Computing t = {t:.2f} with {steps_for_step*it} steps")

            evolution = []
            for idx in np.arange(0, self.rows * self.columns, 2):
                evolution.append(
                    self.step(
                        qubits[idx],
                        qubits[idx + 1],
                        t=t,
                        J=J,
                        v=v,
                        reps=steps_for_step * it,
                    )
                )
            for idx in np.arange(1, self.rows * self.columns - 1, 2):
                evolution.append(
                    self.step(
                        qubits[idx],
                        qubits[idx + 1],
                        t=t,
                        J=J,
                        v=v,
                        reps=steps_for_step * it,
                    )
                )
            measures = []
            for idx in range(self.rows * self.columns):
                measures.append(measure(qubits[idx], key=f"q{idx}"))

            circuit = Circuit([*initial, *evolution, *measures])
            print(f"Len: {len(circuit)}")

            result = Result(sample(circuit, repetitions=100))
            self.tot_results.append(result)

    def plot(self):

        tot_up0 = [
            (res.probabilities["q0"][1] + res.probabilities["q0"][3] / 2)
            for res in self.tot_results
        ]
        tot_up1 = [
            (res.probabilities["q1"][1] + res.probabilities["q1"][3] / 2)
            for res in self.tot_results
        ]

        tot_down0 = [
            (res.probabilities["q0"][2] + res.probabilities["q0"][3] / 2)
            for res in self.tot_results
        ]
        tot_down1 = [
            (res.probabilities["q1"][2] + res.probabilities["q1"][3] / 2)
            for res in self.tot_results
        ]

        plt.plot(self.t, tot_up0, "o-", label="N(0)up")
        plt.plot(self.t, tot_down0, "o-", label="N(0)down")

        plt.plot(self.t, tot_up1, "^--", label="N(1)up")
        plt.plot(self.t, tot_down1, "^--", label="N(1)down")

        plt.legend()


# %%
qfh = QuditFermiHubbard()
print("Studied lattice: \n", qfh)

J = -1
v = -1

t = np.arange(0, 4, 1 / 2)

initial = [
    # set qudit 0 to |1> (up spin)
    X_P_ij(0, 1)(qfh.qubits[0]),
    # set qudit 1 to |3> (mixed state)
    X_P_ij(0, 3)(qfh.qubits[1]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v)

qfh.plot()

# %%
