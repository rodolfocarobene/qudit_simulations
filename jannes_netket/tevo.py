import matplotlib.pyplot as plt
import netket as nk
import numpy as np
from netket import experimental as nkx
from netket.experimental.dynamics import RK45 as RK
from src.exact import ExactDynamics, NonVariationalVectorState

TOTAL_RESULTS = {"ham": [], "ekin": [], "epot": [], "nums": []}

Lx, Ly = 2, 3
thop = -1  # tunneling/hopping
U = 0.5  # coulomb
NUM_FERMIONS = 2

# create the graph our fermions can hop on
g = nk.graph.Grid([Lx, Ly], pbc=False)
n_sites = g.n_nodes

# create a hilbert space with 2 up and 2 down spins
hi = nkx.hilbert.SpinOrbitalFermions(n_sites, n_fermions=NUM_FERMIONS)


def c(site):
    return nkx.operator.fermion.destroy(hi, site)


def cdag(site):
    return nkx.operator.fermion.create(hi, site)


def nc(site):
    return nkx.operator.fermion.number(hi, site)


ekin = 0.0
epot = 0.0
for u, v in g.edges():
    ekin += thop * (cdag(u) * c(v) + cdag(v) * c(u))
    epot += U * nc(u) * nc(v)

ham = ekin + epot

obs = {
    "ekin": ekin,
    "epot": epot,
}
for idx, s in enumerate(g.sites):
    obs[f"n{idx}"] = nc(idx)

# make the initial state
occupations = [
    [1, 1, 0, 0, 0, 0],  # A
    [1, 0, 1, 0, 0, 0],  # B
]

occupation = np.stack(occupations, axis=0)

idx = hi.states_to_numbers(occupation)
vector = np.zeros((hi.n_states,), dtype=np.complex128)
coeffs = [1, 1]  # 0.47430418 + 0.48681048j, 0.64455866 + 0.59635361j]
for i, j in enumerate(idx):
    vector[j] = coeffs[i]
vector /= np.linalg.norm(vector)

vs = NonVariationalVectorState(hi, vector)

print()
for i in range(len(g.sites)):
    exp_n = vs.expect(nc(i)).mean
    if exp_n != 0:
        print(f"N({i}): {exp_n}")

print("\nEkin:", np.round(vs.expect(ekin).mean, 4))
print("Epot:", np.round(vs.expect(epot).mean, 4))
print("Ham:", np.round(vs.expect(ham).mean, 4))


T = 0.4
dt = T / 2

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
plt.savefig("hamiltonian_evolution.pdf")

plt.figure()

result_nums = TOTAL_RESULTS["nums"]
for idx in range(len(g.sites)):
    ev = [result[idx] for result in result_nums]
    plt.plot(times, ev, label=f"n{idx}")
plt.legend()
plt.savefig("numbers_evolution.pdf")

print("\n", np.round(TOTAL_RESULTS["nums"][-1], 4))
