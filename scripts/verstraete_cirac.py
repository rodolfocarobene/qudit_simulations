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

# %%
import matplotlib.pyplot as plt
import numpy as np
from cirq import Circuit, LineQid, Simulator, measure, sample
from logger import log
from primitives import *
from scipy.linalg import expm

simulator = Simulator()


# %% [markdown]
# ## Results


# %%
class QuditResult:
    def __init__(self, res, exact=True):

        # self.physical_qubits = self.qubits[:len(self.qubits)//2]

        if exact:
            self.init_exact(res)
        else:
            self.init_shots(res)

    def init_exact(self, res):

        self.sites = {}

        s = np.abs(res.final_state_vector)
        s = s * s / np.sum(s * s)

        shape = tuple([4] * len(res.qubit_map))
        s = s.reshape(shape)

        for idx, qubit in enumerate(res.qubit_map):
            index = [slice(None)] * s.ndim
            index[idx] = 1
            sum_1 = np.sum(s[tuple(index)])
            index = [slice(None)] * s.ndim
            index[idx] = 3
            sum_3 = np.sum(s[tuple(index)])
            self.sites[idx] = sum_1 + sum_3 / 2

        # self.sites = {idx: self.sites[idx*2] for idx in range(len(self.sites)//2)}

    def init_shots(self, res):

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


# %% [markdown]
# ## Hamiltonian components

# %%
# on-site term, how does it behave???


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


# %%
# Hopping term


class Hhop(Gate):
    def __init__(self, J=1, t=0.1):
        self.J = J
        self.t = t
        super()

    def _qid_shape_(self):
        return (4, 4, 4, 4)

    def _unitary_(self):
        return expm(
            np.array(
                -1j
                * self.t
                * (
                    self.J
                    / (2j)
                    * (-1)
                    * (
                        TensorProduct(
                            sy_gamma_2 * sy_gamma_5,
                            sy_gamma_1 * sy_gamma_5,
                            sy_gamma_5 * sy_gamma_1,
                            sy_gamma_2,
                        )
                        - TensorProduct(
                            sy_gamma_1 * sy_gamma_5,
                            sy_gamma_2 * sy_gamma_5,
                            sy_gamma_5 * sy_gamma_1,
                            sy_gamma_2,
                        )
                        + TensorProduct(
                            sy_gamma_4 * sy_gamma_5,
                            sy_gamma_3 * sy_gamma_5,
                            sy_gamma_5 * sy_gamma_3,
                            sy_gamma_4,
                        )
                        - TensorProduct(
                            sy_gamma_3 * sy_gamma_5,
                            sy_gamma_4 * sy_gamma_5,
                            sy_gamma_5 * sy_gamma_3,
                            sy_gamma_4,
                        )
                    )
                ),
                dtype=np.complex128,
            )
        )

    def _circuit_diagram_info_(self, args):
        return ["Hop(k)", "Hop(k+n)", "Hop(k')", "Hop(k'+n')"]


# %%
# Hopping term


class Haux(Gate):
    def __init__(self, t=0.1):
        self.t = t
        super()

    def _qid_shape_(self):
        return (4, 4)

    def _unitary_(self):
        return expm(
            np.array(
                -1j
                * self.t
                * (-1)
                * (
                    TensorProduct(sy_gamma_1 * sy_gamma_5, sy_gamma_2)
                    + TensorProduct(sy_gamma_3 * sy_gamma_5, sy_gamma_4)
                )
                * (
                    TensorProduct(sy_gamma_5 * sy_gamma_2, sy_gamma_1)
                    + TensorProduct(sy_gamma_5 * sy_gamma_4, sy_gamma_3)
                ),
                dtype=np.complex128,
            )
        )

    def _circuit_diagram_info_(self, args):
        return ["Haux(x)", "Haux(y)"]


# %%
class QuditFermiHubbard:
    def __init__(self, L=1, M=2):
        """Initialize qudit system, L x M."""
        self.rows = L
        self.columns = M

        self.qubits = LineQid.range(L * M * 2, dimension=4)
        self.physical_qubits = self.qubits[: len(self.qubits) // 2]
        self.auxiliary_qubits = self.qubits[len(self.qubits) // 2 :]

        self.tot_results = []

    def __str__(self):
        rows = self.rows
        columns = self.columns

        final_str = ""
        for row in range(rows):
            for col in range(columns):
                if row % 2 == 0:
                    final_str += f"  {row * columns + (col+1) - 1}  "
                else:
                    final_str += f"  {(row+1) * columns - (col) - 1}  "
                if col != columns - 1:
                    final_str += " - "
            final_str += "\n"
            if row != rows - 1:
                for col in range(columns):
                    final_str += f"   |   "
            final_str += "\n"

        final_str += "\n\n"
        for row in range(rows):
            for col in range(columns):
                if row % 2 == 0:
                    final_str += f"  {row * columns + (col+1) - 1}'  "
                else:
                    final_str += f"  {(row+1) * columns - (col) - 1}'  "
                if col != columns - 1:
                    final_str += "- "

            final_str += "\n"
            if row != rows - 1:
                for col in range(columns):
                    final_str += f"  |   "
            final_str += "\n"

        return final_str

    def __repr__(self):
        return self.__str__()

    def print(self):
        print(self.__str__())

    def evolve(self, initial, temps, steps_for_step=10, J=-1, v=0, repetitions=1000):
        exact = repetitions == 0
        if exact:
            if len(initial) != len(self.physical_qubits * 2):
                log.error("In an exact simulation, all qubits must be initialized!")

        v = v / 4
        log.warning(
            "For some reason, v = v / 4 is a required correction. I applied it for you"
        )

        self.tot_results = []
        self.t = temps

        for it, t in enumerate(temps):
            log.info(f"Computing t = {t:.2f} with {steps_for_step*it} steps")

            evolution_circuit = []

            if t != 0:
                tau = t / (steps_for_step * it)

                for _ in range(steps_for_step * it):
                    first = _ == 0 and it == 1

                    for qubit in self.physical_qubits:
                        if first:
                            print(f"On site: {qubit}")

                        # on-site part
                        evolution_circuit.append(Hint(v, tau)(qubit))

                    # hopping terms
                    couples = [(0, 1), (1, 2), (2, 3), (3, 0)]
                    for idx, jdx in couples:
                        if first:
                            print(f"Hop {idx} {jdx} and primes")
                        evolution_circuit.append(
                            Hhop(J, tau)(
                                self.physical_qubits[idx],
                                self.physical_qubits[jdx],
                                self.auxiliary_qubits[idx],
                                self.auxiliary_qubits[jdx],
                            )
                        )

                    # auxiliary ham terms
                    couples = [(0, 3), (3, 2), (2, 1), (1, 0)]
                    for idx, jdx in couples:
                        if first:
                            print(f"Aux {idx} - {jdx}")
                        evolution_circuit.append(
                            Haux(tau)(
                                self.auxiliary_qubits[idx],
                                self.auxiliary_qubits[jdx],
                            )
                        )
            measures = []
            for idx, qubit in enumerate(self.physical_qubits):
                measures.append(measure(qubit, key=f"q{idx}"))

            meas_circuit = Circuit([*initial, *evolution_circuit, *measures])
            circuit = Circuit([*initial, *evolution_circuit])
            # log.info(f"Len: {len(circuit)}")

            if exact:
                # print(circuit)
                result = QuditResult(simulator.simulate(circuit), exact=exact)
            else:
                result = QuditResult(
                    sample(meas_circuit, repetitions=repetitions), exact=exact
                )
            self.tot_results.append((t, result))

    def plot(self, sites=None):

        if sites is None:
            sites = self.tot_results[0][1].sites

        physical = 0
        for site in sites:
            tot_site = [res[1].sites[site] for res in self.tot_results]
            if physical < self.rows * self.columns:
                label = f"N({site}) up"
                linestyle = "solid"
                physical += 1
            else:
                label = f"N({site-physical})' up"
                linestyle = "dotted"
            plt.plot(self.t, tot_site, "o", label=label, linestyle=linestyle)

        plt.legend()


# %%

# %%
qfh = QuditFermiHubbard(2, 2)
print(f"Studied lattice: \n{qfh}")

# %% [markdown]
# ## Example: single fermion

# %%
J = -1
v = 0
repetitions = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.physical_qubits[0]),
    X_P_ij(0, 1, 0)(qfh.physical_qubits[1]),
    X_P_ij(0, 1, 0)(qfh.physical_qubits[2]),
    X_P_ij(0, 1, 0)(qfh.physical_qubits[3]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[0]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[1]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[2]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[3]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v, repetitions=repetitions)

qfh.plot()

# %% [markdown]
# ## Two near fermions

# %%
J = -1
v = 0
repetitions = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.physical_qubits[0]),
    X_P_ij(0, 1)(qfh.physical_qubits[1]),
    X_P_ij(0, 1, 0)(qfh.physical_qubits[2]),
    X_P_ij(0, 1, 0)(qfh.physical_qubits[3]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[0]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[1]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[2]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[3]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v, repetitions=repetitions)

qfh.plot()

# %% [markdown]
# ## Three fermions

# %%
J = -1
v = 0
repetitions = 0

t = np.arange(0, 2, 1 / 2)

initial = [
    X_P_ij(0, 1)(qfh.physical_qubits[0]),
    X_P_ij(0, 1, 0)(qfh.physical_qubits[1]),
    X_P_ij(0, 1)(qfh.physical_qubits[2]),
    X_P_ij(0, 1)(qfh.physical_qubits[3]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[0]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[1]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[2]),
    X_P_ij(0, 1, 0)(qfh.auxiliary_qubits[3]),
]
qfh.evolve(initial, t, steps_for_step=10, J=J, v=v, repetitions=repetitions)

qfh.plot()

# %%
a = (
    TensorProduct(sy_gamma_1 * sy_gamma_5, sy_gamma_2)
    + TensorProduct(sy_gamma_3 * sy_gamma_5, sy_gamma_4)
) * (
    TensorProduct(sy_gamma_5 * sy_gamma_2, sy_gamma_1)
    + TensorProduct(sy_gamma_5 * sy_gamma_4, sy_gamma_3)
)
a

# %%
b = (
    TensorProduct(sy_gamma_1 * sy_gamma_5, sy_gamma_2)
    + TensorProduct(sy_gamma_3 * sy_gamma_5, sy_gamma_4)
) * (
    TensorProduct(sy_gamma_2, sy_gamma_1 * sy_gamma_5)
    + TensorProduct(sy_gamma_4, sy_gamma_3 * sy_gamma_5)
)

b

# %%
a == -b

# %%
