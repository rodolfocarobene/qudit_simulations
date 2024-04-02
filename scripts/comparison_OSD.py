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
# ### Comparison of exact ququart exponentiation and gates method

from cirq import LineQid
from primitives import *

# %%
from sympy import exp

t = 1
J = -1

N = 2  # number of sites

qubits = LineQid.range(N, dimension=4)

# %% [markdown]
# #### U1

# %%
exact = exp(
    -1j * t * (J / (2j) * (TensorProduct(Matrix(sy_gamma_2 * sy_gamma_5), sy_gamma_1)))
)
exact.simplify().simplify()

# %%
tau = t * J
a = [
    Matrix(UCSUM()(qubits[0], qubits[1])._unitary_()),
    TensorProduct(Matrix(X_P_ij(0, 2, tau)(qubits[0])._unitary_()), sy_id),
    TensorProduct(Matrix(X_P_ij(1, 3, -tau)(qubits[0])._unitary_()), sy_id),
    TensorProduct(Matrix(Y_P_ij(0, 2, -np.pi)(qubits[0])._unitary_()), sy_id),
    Matrix(UCSUM()(qubits[0], qubits[1])._unitary_()),
    TensorProduct(Matrix(Y_P_ij(0, 2, np.pi)(qubits[0])._unitary_()), sy_id),
]

gates = 1
for s in a:
    gates = gates * s
gates.as_immutable().simplify().simplify()

# %%
gates == exact

# %% [markdown]
# #### U2

# %%
exact = exp(
    -1j * t * (J / (2j) * (-TensorProduct(Matrix(sy_gamma_1 * sy_gamma_5), sy_gamma_2)))
)
exact.simplify().simplify()

# %%
tau = t * J
a = [
    TensorProduct(Matrix(Z_P_ij(0, 2, np.pi / 2)(qubits[0])._unitary_()), sy_id),
    TensorProduct(Matrix(Z_P_ij(1, 3, -np.pi / 2)(qubits[0])._unitary_()), sy_id),
    TensorProduct(sy_id, Matrix(Z_P_ij(0, 2, np.pi / 2)(qubits[1])._unitary_())),
    TensorProduct(sy_id, Matrix(Z_P_ij(1, 3, np.pi / 2)(qubits[1])._unitary_())),
    UCSUM()(qubits[0], qubits[1])._unitary_(),
    TensorProduct(Matrix(X_P_ij(0, 2, tau)(qubits[0])._unitary_()), sy_id),
    TensorProduct(Matrix(X_P_ij(1, 3, tau)(qubits[0])._unitary_()), sy_id),
    TensorProduct(Matrix(Y_P_ij(0, 2, -np.pi)(qubits[0])._unitary_()), sy_id),
    UCSUM()(qubits[0], qubits[1])._unitary_(),
    TensorProduct(Matrix(Y_P_ij(0, 2, np.pi)(qubits[0])._unitary_()), sy_id),
    TensorProduct(Matrix(Z_P_ij(1, 3, np.pi / 2)(qubits[0])._unitary_()), sy_id),
    TensorProduct(Matrix(Z_P_ij(0, 2, -np.pi / 2)(qubits[0])._unitary_()), sy_id),
    TensorProduct(sy_id, Matrix(Z_P_ij(1, 3, -np.pi / 2)(qubits[1])._unitary_())),
    TensorProduct(sy_id, Matrix(Z_P_ij(0, 2, -np.pi / 2)(qubits[1])._unitary_())),
]

gates = 1
for s in a:
    gates = gates * s
gates.as_immutable().simplify().simplify()

# %%
gates == exact

# %%
