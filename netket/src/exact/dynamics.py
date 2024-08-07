from netket.experimental.driver.tdvp_common import TDVPBaseDriver, odefun
from netket.experimental.dynamics import RKIntegratorConfig
from netket.hilbert import AbstractHilbert
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.utils.types import Array, Callable, Union
from netket.vqs import VariationalState

from .state import NonVariationalVectorState
from .utils import to_matrix


class ExactDynamics(TDVPBaseDriver):
    def __init__(
        self,
        hilbert: AbstractHilbert,
        operator: AbstractOperator,
        vector_array: Union[Array, VariationalState],
        integrator: RKIntegratorConfig,
        *,
        t0: float = 0.0,
        propagation_type: str = "real",
        error_norm: Union[str, Callable] = "euclidean",
        sparse: bool = True,
    ):
        if isinstance(vector_array, NonVariationalVectorState):
            variational_state = vector_array
        else:
            variational_state = NonVariationalVectorState(hilbert, vector_array)

        self.propagation_type = propagation_type
        if propagation_type == "real":
            self._dynamics_factor = 1j
            variational_state.to_complex()
        elif propagation_type == "imag":
            self._dynamics_factor = 1
        else:
            raise ValueError("propagation_type must be one of 'real', 'imag'")

        self._operator_matrix = None
        self._sparse = sparse
        self._time_independent_op = isinstance(operator, AbstractOperator)

        super().__init__(
            operator, variational_state, integrator, t0=t0, error_norm=error_norm
        )

        if self._time_independent_op:
            self.operator_matrix(0.0)

    def operator_matrix(self, t):
        if self._time_independent_op:
            if self._operator_matrix is None:
                op = self._generator(t)
                # we cache it here instead
                self._operator_matrix = to_matrix(op, sparse=self._sparse, cache=False)
            return self._operator_matrix
        else:
            op = self._generator(t)
            return to_matrix(op, sparse=self._sparse, cache=False)


@odefun.dispatch
def odefun_tdvp(  # noqa: F811
    state: NonVariationalVectorState, driver: ExactDynamics, t, w, *, stage=0
):
    # pylint: disable=protected-access
    # the output of this is now parameters = a full Psi
    #
    state.parameters = w
    state.reset()

    H_t = driver.operator_matrix(t)

    # dPsi/dt =  - factor (H-E) Psi = f(t)
    HPsi, E = _Opsi_and_expO(H_t, state.vector)
    driver._loss_stats = Stats(mean=E, error_of_mean=0.0)
    dPsi_dt = _schrodinger(HPsi, E, state.vector, driver._dynamics_factor)
    # If parameters are real, then take only real part of the gradient (if it's complex)
    driver._dw = {"vector": dPsi_dt}

    if stage == 0:
        # save the info at the initial point
        driver._ode_data = {
            "loss_stats": driver._loss_stats,
        }

    return driver._dw


def _schrodinger(Hpsi, E, psi, factor):
    return -factor * (Hpsi - E * psi)


def _Opsi_and_expO(O, v):
    Opsi = O @ v
    expval = (v.conj() * Opsi).sum()
    return Opsi, expval
