import matplotlib.pyplot as plt
import netket as nk
import numpy as np
from netket import experimental as nkx
from netket.experimental.dynamics import RK23
from src.exact import ExactDynamics, NonVariationalVectorState

TOTAL_RESULTS = {"ham": [], "ekin": [], "epot": [], "nums": []}

Lx, Ly = 2, 2
thop = -1  # tunneling/hopping
U = 0.5  # coulomb

# create the graph our fermions can hop on
g = nk.graph.Grid([Lx, Ly], pbc=False)
n_sites = g.n_nodes

# create a hilbert space with 2 up and 2 down spins
hi = nkx.hilbert.SpinOrbitalFermions(n_sites, n_fermions=2)


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
    [1, 1, 0, 0],  # A
    [1, 0, 0, 1],  # B
    [1, 0, 1, 0],  # C
    [0, 1, 1, 0],  # D
    [0, 1, 0, 1],  # E
    [0, 0, 1, 1],  # F
]

occupation = np.stack(occupations, axis=0)

idx = hi.states_to_numbers(occupation)
vector = np.zeros((hi.n_states,), dtype=np.complex128)
coeffs = [
    0.87806136 + 0.47854806j,
    0.89290143 + 0.45025218j,
    0.14688749 + 0.98915321j,
    0.82658314 + 0.56281463j,
    0.59238681 + 0.80565369j,
    0.84171922 + 0.5399155j,
]
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


T = 0.1
dt = 0.025

integrator = RK23(dt=dt)
te = ExactDynamics(
    hi,
    ekin + epot,
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

    energy_str = f"t = {t:.2}\t\t"
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

times = np.arange(0, T + dt, dt)

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
