# ## Circ example

# +
import cirq
import numpy as np


class QutritPlusGate(cirq.Gate):
    """A gate that adds one in the computational basis of a qutrit.

    This gate acts on three-level systems. In the computational basis of
    this system it enacts the transformation U|x〉 = |x + 1 mod 3〉, or
    in other words U|0〉 = |1〉, U|1〉 = |2〉, and U|2> = |0〉.
    """

    def _qid_shape_(self):
        # By implementing this method this gate implements the
        # cirq.qid_shape protocol and will return the tuple (3,)
        # when cirq.qid_shape acts on an instance of this class.
        # This indicates that the gate acts on a single qutrit.
        return (3,)

    def _unitary_(self):
        # Since the gate acts on three level systems it has a unitary
        # effect which is a three by three unitary matrix.
        return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return "[+1]"


# Here we create a qutrit for the gate to act on.
q0 = cirq.LineQid(0, dimension=3)

# We can now enact the gate on this qutrit.
circuit = cirq.Circuit(QutritPlusGate().on(q0))

# When we print this out we see that the qutrit is labeled by its dimension.
print(circuit)

# -

# cirq.Qid is the type that represents both qubits and qudits. Different options exist:
#
# - NamedQid
# - GridQid
# - Lineqid
#
# An interesting function may be: circ.qid_shape

# +
# Create an instance of the qutrit gate defined above.
gate = QutritPlusGate()

# Verify that it acts on a single qutrit.
print(cirq.qid_shape(gate))
# -

# The magic methods _unitary_, _apply_unitary_, _mixture_, and _kraus_ can be used to define unitary gates, mixtures, and channels can be used with qudits (see protocols for how these work.)

# +
# Create an instance of the qutrit gate defined above. This gate implements _unitary_.
gate = QutritPlusGate()

# Because it acts on qutrits, its unitary is a 3 by 3 matrix.
print(cirq.unitary(gate))
# -

# Cirq's simulators can be used to simulate or sample from circuits which act on qudits.
#
# Simulators like cirq.Simulator and cirq.DensityMatrixSimulator will return simulation results with larger states than the same size qubit circuit when simulating qudit circuits.
# The size of the state returned is determined by the product of the dimensions of the qudits being simulated.
# For example, the state vector output of cirq.Simulator after simulating a circuit on a qubit, a qutrit, and a qutrit will have 2 * 3 * 3 = 18 elements.
# You can call cirq.qid_shape(simulation_result) to check the qudit dimensions.

# +
# Create a circuit from the gate we defined above.
q0 = cirq.LineQid(0, dimension=3)
circuit = cirq.Circuit(QutritPlusGate()(q0))

# Run a simulation of this circuit.
sim = cirq.Simulator()
result = sim.simulate(circuit)

# Verify that the returned state is that of a qutrit.
print(cirq.qid_shape(result))

# Create a circuit with three qutrit gates.
q0, q1 = cirq.LineQid.range(2, dimension=3)
circuit = cirq.Circuit(
    [
        QutritPlusGate()(q0),
        QutritPlusGate()(q1),
        QutritPlusGate()(q1),
        cirq.measure(q0, q1, key="x"),
    ]
)

# Sample from this circuit.
result = cirq.sample(circuit, repetitions=3)

# See that the results are all integers from 0 to 2.
print(result)
# -
