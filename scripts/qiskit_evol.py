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
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import BoundaryCondition, LineLattice
from qiskit_nature.second_q.mappers import JordanWignerMapper

# %% [markdown]
# ### Helpers


# %%
def evolve(temps, steps_for_step=10, J=1, v=0):

    fhm = FermiHubbardModel(
        line_lattice.uniform_parameters(
            uniform_interaction=J,
            uniform_onsite_potential=0.0,
        ),
        onsite_interaction=v,
    )

    tot_results = []

    for it, t in enumerate(temps):
        print(f"Computing t = {t:.2f} with {steps_for_step*(it+1)} steps")

        mapper = JordanWignerMapper()
        ham = mapper.map(fhm.second_q_op())

        evol_gate = PauliEvolutionGate(
            ham, time=t, synthesis=SuzukiTrotter(reps=steps_for_step * (it + 1))
        )

        evolved_state = QuantumCircuit(num_nodes * 2)

        evolved_state.x(0)  # set first node as up
        evolved_state.x(2)  # and second half-filled
        evolved_state.x(3)

        evolved_state.append(evol_gate, range(num_nodes * 2))

        result = Statevector(evolved_state).probabilities_dict()
        tot_results.append(result)
    return tot_results


# %% [markdown]
# ### Example

# %%
num_nodes = 2

boundary_condition = BoundaryCondition.OPEN
line_lattice = LineLattice(num_nodes=num_nodes, boundary_condition=boundary_condition)
line_lattice.draw()

# %%
t = -1.0  # the interaction parameter
v = 1.0  # the onsite potential
u = 0.0  # the interaction parameter U

fhm = FermiHubbardModel(
    line_lattice.uniform_parameters(
        uniform_interaction=t,
        uniform_onsite_potential=v,
    ),
    onsite_interaction=u,
)

mapper = JordanWignerMapper()
ham = mapper.map(fhm.second_q_op())

evol_gate = PauliEvolutionGate(ham, time=1.5, synthesis=SuzukiTrotter(reps=40))

evolved_state = QuantumCircuit(num_nodes * 2)
evolved_state.x(0)
evolved_state.x(2)
evolved_state.append(evol_gate, range(num_nodes * 2))

# %%
Statevector(evolved_state).probabilities_dict()

# %% [markdown]
# ### Evolution

# %%
J = -1
v = 1

t = np.arange(0, 5, 1 / 2)
results = evolve(t, steps_for_step=10, J=J, v=v)

# %%
results[0]

# %%
tot_up0 = []
tot_up1 = []
tot_down0 = []
tot_down1 = []

for res in results:
    up0 = 0
    down0 = 0
    up1 = 0
    down1 = 0

    void0 = 0
    void1 = 0

    for key in res:
        if key[3] == "1":
            up0 += res[key]
        if key[2] == "1":
            down0 += res[key]
        if key[3] != "1" and key[2] != "1":
            void0 += res[key]

        if key[1] == "1":
            up1 += res[key]
        if key[0] == "1":
            down1 += res[key]
        if key[1] != "1" and key[0] != "1":
            void1 += res[key]

    tot_up0.append(up0 / (up0 + down0 + void0))
    tot_down0.append(down0 / (up0 + down0 + void0))
    tot_up1.append(up1 / (up1 + down1 + void1))
    tot_down1.append(down1 / (up1 + down1 + void1))

# %%
plt.plot(t, tot_up0, "o-", label="N(0)up")
plt.plot(t, tot_down0, "o-", label="N(0)down")

plt.plot(t, tot_up1, "^--", label="N(1)up")
plt.plot(t, tot_down1, "^--", label="N(1)down")

plt.legend()

plt.savefig("plots/4qubits.pdf")

# %%
