## Primitives

### Pauli Matrices (2x2)

These are standard 2x2 Pauli matrices used in quantum mechanics.

- `sx`: Pauli X matrix.
- `sy`: Pauli Y matrix.
- `sz`: Pauli Z matrix.
- `si`: Identity matrix.

### Gamma Matrices (4x4)

These matrices are used to represent extended operations in a 4-dimensional
space.

- `sy_gamma_1`: Tensor product of `sx` and `si`.
- `sy_gamma_2`: Tensor product of `sy` and `si`.
- `sy_gamma_3`: Tensor product of `sz` and `sx`.
- `sy_gamma_4`: Tensor product of `sz` and `sy`.
- `sy_gamma_5`: Tensor product of `sz` and `sz`.
- `sy_id`: Identity matrix (4x4).

### Bra-Kets for Ququarts

These vectors represent the basis states of a 4-level quantum system.

- `ket_0`: Basis state |0>.
- `bra_0`: Dual of `ket_0`.
- `ket_1`: Basis state |1>.
- `bra_1`: Dual of `ket_1`.
- `ket_2`: Basis state |2>.
- `bra_2`: Dual of `ket_2`.
- `ket_3`: Basis state |3>.
- `bra_3`: Dual of `ket_3`.

### Two-Qudit Gate Definition

This gate represents an extended CNOT gate for ququarts.

- `X`: Combination of bra-kets for state transitions.
- `sy_ucsum`: A composite gate defined using tensor products and `X`.
- `sy_ad_ucsum`: Adjoint of `sy_ucsum`.

### Custom Cirq Gates

These are custom gates defined for specific operations on ququarts.

#### Rotation Gates

- **X_P_ij**: Rotation around X for the (i, j) states.
- **Y_P_ij**: Rotation around Y for the (i, j) states.
- **Z_P_ij**: Rotation around Z for the (i, j) states.

#### Gamma Gates

- **Gamma1**: Gate representing `sy_gamma_1`.
- **Gamma2**: Gate representing `sy_gamma_2`.
- **Gamma3**: Gate representing `sy_gamma_3`.
- **Gamma4**: Gate representing `sy_gamma_4`.
- **Gamma5**: Gate representing `sy_gamma_5`.

#### Identity Gate

- **Id**: Identity gate for 4-level systems.

#### UCSUM Gates

- **UCSUM**: Composite gate `sy_ucsum`.
- **UCSUMDag**: Adjoint of the `UCSUM` gate.
