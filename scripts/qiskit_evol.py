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
# ## Fermi-Hubbard evolution using Qiskit

# %%
import matplotlib.pyplot as plt
import numpy as np
from logger import log
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    LatticeDrawStyle,
    LineLattice,
    SquareLattice,
)
from qiskit_nature.second_q.mappers import JordanWignerMapper

# %% [markdown]
# ### Helpers


# %%
class QubitResult:
    def __init__(self, res):

        self.sites = {}

        num_qubits = len(list(res)[0]) // 2

        for qubit in range(num_qubits):
            up = 0
            down = 0
            void = 0

            for key in res:
                upidx = -1 - 2 * qubit
                downidx = -2 - 2 * qubit
                up_or_down = False
                if key[upidx] == "1":
                    up += res[key]
                    up_or_down = True
                if key[downidx] == "1":
                    up_or_down = True
                    down += res[key]
                if up_or_down is False:
                    void += res[key]

            self.sites[qubit] = up / (up + down + void)


# %%
class QubitFermiHubbard:
    def __init__(self, L=1, M=2):
        """Initialize qudit system, L x M."""
        self.rows = L
        self.columns = M

        boundary_condition = BoundaryCondition.OPEN
        if L == 1:
            self.lattice = LineLattice(L * M, boundary_condition=boundary_condition)
        else:
            self.lattice = SquareLattice(M, L, boundary_condition=boundary_condition)
        self.tot_results = []

    def evolve(self, initial, temps, steps_for_step=10, J=1, v=0):

        self.t = temps

        fhm = FermiHubbardModel(
            self.lattice.uniform_parameters(
                uniform_interaction=J,
                uniform_onsite_potential=v,
            ),
            onsite_interaction=v,
        )
        self.tot_results = []
        for it, t in enumerate(temps):
            log.info(f"Computing t = {t:.2f} with {steps_for_step*(it)} steps")

            mapper = JordanWignerMapper()
            ham = mapper.map(fhm.second_q_op())

            evolved_state = QuantumCircuit(self.rows * self.columns * 2)

            for idx, init in enumerate(initial):
                if init == "1":
                    evolved_state.x(idx)

            if it > 0:
                evol_gate = PauliEvolutionGate(
                    ham, time=t, synthesis=SuzukiTrotter(reps=steps_for_step * (it))
                )
                evolved_state.append(evol_gate, range(self.rows * self.columns * 2))

            log.info(
                f"Len: {evolved_state.decompose().decompose().decompose().depth()}"
            )

            result = Statevector(evolved_state).probabilities_dict()
            self.tot_results.append(QubitResult(result))

    def plot(self, sites=None):
        results = self.tot_results
        t = self.t

        if sites is None:
            sites = results[0].sites

        for site in sites:
            tot_site = [res.sites[site] for res in results]
            plt.plot(t, tot_site, "o-", label=f"N({site}) up")
        plt.legend()


# %% [markdown]
# ### Example

# %%
qfh = QubitFermiHubbard(2, 2)
qfh.lattice.draw(style=LatticeDrawStyle(with_labels=True))

# %%
J = -1
v = 0

t = np.arange(0, 3, 1 / 2)

qfh.evolve("10110100", t, steps_for_step=10, J=J, v=v)

qfh.plot()

# %%
