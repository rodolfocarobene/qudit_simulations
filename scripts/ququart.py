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
N = 2  # number of sites

qubits = LineQid.range(N, dimension=4)
qubits

# %% [markdown]
# ## Manual implementation


# %%
def step(qubit0, qubit1, t, J, v, reps):
    def single(squbit0, squbit1, tau, sJ, sv):
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
            X_P_ij(0, 2, -np.pi / 2)(qubits[1]),
            X_P_ij(1, 3, np.pi / 2)(qubits[1]),
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
            X_P_ij(1, 3, -np.pi / 2)(qubits[1]),
            X_P_ij(0, 2, np.pi / 2)(qubits[1]),
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
        a += single(qubit0, qubit1, t, J, v)
    return a


def evolve(initial, temps, steps_for_step=10, J=-1, v=0):
    tot_results = []

    for it, t in enumerate(temps):
        print(f"Computing t = {t:.2f} with {steps_for_step*(it+1)} steps")
        circuit = Circuit(
            [
                *initial,
                step(
                    qubits[0], qubits[1], t=t, J=J, v=v, reps=steps_for_step * (it + 1)
                ),
                measure(qubits[0], key="q0"),
                measure(qubits[1], key="q1"),
            ]
        )

        result = Result(sample(circuit, repetitions=100))
        tot_results.append(result)
    return tot_results


# %%
J = -1
v = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    # set qudit 0 to |1> (up spin)
    X_P_ij(0, 1)(qubits[0]),
    # set qudit 1 to |3> (mixed state)
    X_P_ij(0, 3)(qubits[1]),
]
results = evolve(initial, t, steps_for_step=10, J=J, v=v)

# +
tot_up0 = [
    (res.probabilities["q0"][1] + res.probabilities["q0"][3] / 2) for res in results
]
tot_up1 = [
    (res.probabilities["q1"][1] + res.probabilities["q1"][3] / 2) for res in results
]

tot_down0 = [
    (res.probabilities["q0"][2] + res.probabilities["q0"][3] / 2) for res in results
]
tot_down1 = [
    (res.probabilities["q1"][2] + res.probabilities["q1"][3] / 2) for res in results
]
1
plt.plot(t, tot_up0, "o-", label="N(0)up")
plt.plot(t, tot_down0, "o-", label="N(0)down")

plt.plot(t, tot_up1, "^--", label="N(1)up")
plt.plot(t, tot_down1, "^--", label="N(1)down")

plt.legend()


# %% [markdown]
# ## Trotterized implementation


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
def evolve(initial, temps, steps_for_step=10, J=-1, v=0):
    tot_results = []

    for it, t in enumerate(temps):
        print(f"Computing t = {t:.2f} with {steps_for_step*(it+1)} steps")
        circuit = Circuit(
            [
                *initial,
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


# %%
# +
J = -1
v = 0

t = np.arange(0, 4, 1 / 2)

initial = [
    # set qudit 0 to |1> (up spin)
    X_P_ij(0, 1)(qubits[0]),
    # set qudit 1 to |3> (mixed state)
    X_P_ij(0, 3)(qubits[1]),
]
results = evolve(initial, t, steps_for_step=10, J=J, v=v)

# +
tot_up0 = [
    (res.probabilities["q0"][1] + res.probabilities["q0"][3] / 2) for res in results
]
tot_up1 = [
    (res.probabilities["q1"][1] + res.probabilities["q1"][3] / 2) for res in results
]

tot_down0 = [
    (res.probabilities["q0"][2] + res.probabilities["q0"][3] / 2) for res in results
]
tot_down1 = [
    (res.probabilities["q1"][2] + res.probabilities["q1"][3] / 2) for res in results
]
1
plt.plot(t, tot_up0, "o-", label="N(0)up")
plt.plot(t, tot_down0, "o-", label="N(0)down")

plt.plot(t, tot_up1, "^--", label="N(1)up")
plt.plot(t, tot_down1, "^--", label="N(1)down")

plt.legend()

# %%
