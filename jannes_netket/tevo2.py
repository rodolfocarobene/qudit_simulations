# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import json

import matplotlib.pyplot as plt

# +
import netket as nk
import numpy as np
from netket import experimental as nkx

Lx, Ly = 2, 4
thop = 1  # tunneling/hopping
U = 0.01  # coulomb

# create the graph our fermions can hop on
g = nk.graph.Grid([Lx, Ly], pbc=True)
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
    ekin += -thop * cdag(u) * c(v) - thop * cdag(v) * c(u)
    epot += U * nc(u) * nc(v)

ham = ekin + epot

obs = {
    "n0": nc(0),
    "n1": nc(1),
    "n2": nc(2),
    "nn": nc(0) * nc(1),
    "ekin": ekin,
    "epot": epot,
}

print("Hamiltonian =", ham.operator_string())

# make the initial state
occupation1 = np.zeros(hi.size, dtype="bool")
occupation1[0] = True
occupation1[1] = True
occupation2 = np.zeros(hi.size, dtype="bool")
occupation2[1] = True
occupation2[2] = True
occupation3 = np.zeros(hi.size, dtype="bool")
occupation3[2] = True
occupation3[3] = True
occupations = np.stack([occupation1, occupation2, occupation3], axis=0)

idxs = hi.states_to_numbers(occupations)
vector = np.zeros((hi.n_states,), dtype=np.complex128)
vector[idxs[0]] = 0.9
vector[idxs[1]] = -0.5j
vector[idxs[2]] = -1

# other option: check conservatioj of energy
# vector = np.linalg.eigh(ham.to_dense())[1][:,1]
vector /= np.linalg.norm(vector)
v0 = vector.copy()


from netket.experimental.dynamics import RK4
from src.exact import ExactDynamics, NonVariationalVectorState

vs = NonVariationalVectorState(hi, v0.copy())
print("initial energy:", vs.expect(ham))

dt = 1e-2
integrator = RK4(dt=dt)
te = ExactDynamics(
    hi,
    ham,c
    vs,
    integrator,
    t0=0,
    propagation_type="real",
    sparse=True,
)

T = 1.0


def _print_obs(step_nr, log_data, driver):
    # print("log_data = ", log_data)
    return True


te.run(T, out="dynamics_out", show_progress=True, obs=obs, callback=_print_obs)


# +
# do also the expm approach


def taylor_expm(n_order=3):  # order of the error in dt
    Hmat = ham.to_dense()
    print("Hmat = ", Hmat.shape)
    print(dt)
    Hn = np.eye(hi.n_states, dtype=complex)  # order H^n
    coeffn = complex(1)
    exp_ham_dt = Hn

    for n in range(1, n_order):
        coeffn *= -1j * dt / n
        Hn = Hn @ Hmat
        exp_ham_dt = exp_ham_dt + coeffn * Hn
    return exp_ham_dt


exp_ham_dt = taylor_expm(n_order=4)

exp_ham_dt.shape
# -

dt

# +
# import scipy


# scipy.linalg.expm(-1j*dt*Hmat)
# -

exp_ham_dt

obs_mats = {}
for k, o in obs.items():
    obs_mats[k] = o.to_sparse()

# +
import pandas as pd

obs_expm = []
Hmat = ham.to_sparse()

vt = v0.copy()
print("Initial energy", vt.conj().dot(Hmat.dot(vt)))

for t in np.arange(0, T, dt):
    log_data = {
        "t": t,
        "Generator": np.sum(vt.conj().dot(Hmat.dot(vt))).real,
    }
    for k, o in obs_mats.items():
        log_data[k] = np.sum(vt.conj().dot(o.dot(vt))).real
    obs_expm.append(log_data)
    vt = exp_ham_dt.dot(vt)
    vt /= np.linalg.norm(vt)

df = pd.DataFrame(obs_expm)
df

# +
# plot stuff
import json

with open("dynamics_out.log", "r") as f:
    data = json.load(f)
# -

for key in ["Generator"] + list(obs.keys()):
    print(key)
    ts = data["t"]["value"]
    values = data[key]["Mean"]["real"]
    if len(ts) == len(values) + 1:
        ts = ts[1:]
    plt.plot(ts, values, label="Schrodinger")
    plt.plot(df["t"].values, df[key].values, label="Expm", linestyle="--")
    plt.legend()
    # plt.yscale('log')
    plt.savefig("tevo2.pdf")
