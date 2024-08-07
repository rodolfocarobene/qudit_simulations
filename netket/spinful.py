import json

import matplotlib.pyplot as plt
import numpy as np
from src.exact import ExactDynamics, NonVariationalVectorState

import netket as nk
from netket import experimental as nkx
from netket.experimental.dynamics import RK45 as RK

TOTAL_RESULTS = {"ham": [], "ekin": [], "epot": [], "nums": []}
T = 1.5
num_steps = 30
dt = T / num_steps  # time step
TOTAL_RESULTS["temps"] = np.linspace(0, T, num_steps + 1, True).tolist()

Lx, Ly = 2, 3
J = -1  # tunneling/hopping
U = 0.5  # coulomb
NUM_FERMIONS = 4

# make the initial state
occupations = [
    [1, 1, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0] + [1, 0, 0, 0, 1, 0],
]
coeffs = [1, 1]

# create the graph our fermions can hop on
g = nk.graph.Grid([Lx, Ly], pbc=False)
n_sites = g.n_nodes

# create a hilbert space with 2 up and 2 down spins
hi = nkx.hilbert.SpinOrbitalFermions(n_sites, s=0.5, n_fermions=NUM_FERMIONS)


def c(site, sz):
    return nkx.operator.fermion.destroy(hi, site, sz)


def cdag(site, sz):
    return nkx.operator.fermion.create(hi, site, sz)


def nc(site, sz):
    return nkx.operator.fermion.number(hi, site, sz)


ekin = 0.0
epot = 0.0
for u, v in g.edges():
    for sz in [-1, +1]:
        ekin += J * (cdag(u, sz) * c(v, sz) + cdag(v, sz) * c(u, sz))
for u, site in enumerate(g.sites):
    epot += U * nc(u, 1) * nc(u, -1)

ham = ekin + epot

obs = {
    "ekin": ekin,
    "epot": epot,
}
for idx, s in enumerate(g.sites):
    obs[f"n{idx}"] = nc(idx, 1)
    obs[f"n{idx}'"] = nc(idx, -1)


occupation = np.stack(occupations, axis=0)

idx = hi.states_to_numbers(occupation)
vector = np.zeros((hi.n_states,), dtype=np.complex128)
for i, j in enumerate(idx):
    vector[j] = coeffs[i]
vector /= np.linalg.norm(vector)

vs = NonVariationalVectorState(hi, vector)

print()
for i in range(len(g.sites)):
    exp_n = vs.expect(nc(i, 1)).mean
    if exp_n != 0:
        print(f"N({i}): {exp_n}")
    exp_n = vs.expect(nc(i, -1)).mean
    if exp_n != 0:
        print(f"N({i})': {exp_n}")

print("\nEkin:", np.round(vs.expect(ekin).mean, 4))
print("Epot:", np.round(vs.expect(epot).mean, 4))
print("Ham:", np.round(vs.expect(ham).mean, 4))


integrator = RK(dt=dt)
te = ExactDynamics(
    hi,
    ham,
    vs,
    integrator,
    t0=0,
    propagation_type="real",
    sparse=True,
)


def _print_obs(step_nr, log_data, driver):
    t = log_data["t"]
    ekin = float(np.real(log_data["ekin"].mean))
    epot = float(np.real(log_data["epot"].mean))
    h = ekin + epot
    N = 4

    energy_str = f"t = {t:.4}\t\t"
    energy_str += f"Ekin={np.round(ekin, N)}\t"
    energy_str += f"Pot={np.round(epot, N)}\t"
    energy_str += f"H={np.round(h, N)}\t"
    print(energy_str)

    global TOTAL_RESULTS
    TOTAL_RESULTS["ham"].append(h)
    TOTAL_RESULTS["ekin"].append(ekin)
    TOTAL_RESULTS["epot"].append(epot)

    nums = []
    for idx in range(len(g.sites)):
        nums.append(float(np.real(log_data[f"n{idx}"].mean)))
        nums.append(float(np.real(log_data[f"n{idx}'"].mean)))
    TOTAL_RESULTS["nums"].append(nums)

    return True


print("")
te.run(T, out="dynamics_out", show_progress=False, obs=obs, callback=_print_obs)

len_t = len(TOTAL_RESULTS["ekin"])
times = np.arange(0, T + dt, dt)[:len_t]

plt.figure()
plt.plot(times, TOTAL_RESULTS["ekin"], label="Ekin")
plt.plot(times, TOTAL_RESULTS["epot"], label="Epot")
plt.plot(times, TOTAL_RESULTS["ham"], label="Ham")
plt.legend()
plt.savefig("spinful_ham.pdf")

plt.figure()

result_nums = TOTAL_RESULTS["nums"]
for idx in range(len(g.sites) * 2):
    ev = [result[idx] for result in result_nums]
    name = idx // 2
    if idx % 2 == 1:
        name = str(name) + "'"
    plt.plot(times, ev, label=f"n{name}")
plt.legend()
plt.savefig("spinful_num.pdf")

print("\n", np.round(TOTAL_RESULTS["nums"][-1], 4))

with open("spinful_data.json", "w") as file:
    json.dump(TOTAL_RESULTS, file)
