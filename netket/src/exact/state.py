import jax
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401
from jax import numpy as jnp

from netket import jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.utils.types import Array, DType, Union
from netket.vqs.base import VariationalState


class NonVariationalVectorState(VariationalState):
    def __init__(
        self,
        hilbert: AbstractHilbert,
        vector_array: Union[Array, VariationalState],
        *,
        dtype: DType = None,
        normalize: bool = True,
    ):
        super().__init__(hilbert)
        if isinstance(vector_array, VariationalState):
            vector_array = vector_array.to_array()
        self._vector = jnp.asarray(vector_array).ravel()
        if dtype is not None:
            self._vector = self._vector.astype(dtype)
        if normalize:
            self.normalize()
        if not self.hilbert.is_indexable:
            raise Exception("Cannot create state if hilbert space is not indexable.")
        if self.hilbert.n_states != self._vector.size:
            raise Exception(
                "Size of vector does not correspond to number of states in the hilbert space."
            )

    def to_complex(self):
        if not nkjax.is_complex_dtype(self._vector.dtype):
            dtype = nkjax.dtype_complex(self._vector.dtype)
            self._vector = self._vector.astype(dtype)

    def normalize(self):
        self._vector = _normalize(self._vector)

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, other):
        self._normalized = False
        other = jnp.asarray(other).ravel()
        if other.shape != self._vector.shape:
            raise ValueError("Provided vector does not match internal vector in shape.")
        # make sure we never down cast
        _dtype = jnp.promote_types(self._vector.dtype, other.dtype)
        self._vector = other.astype(_dtype)

    @property
    def normalized_vector(self):
        return _normalize(self._vector)

    @property
    def parameters(self):
        return {"vector": self.vector}

    @parameters.setter
    def parameters(self, other):
        self.vector = other["vector"]

    def norm(self):
        return _compute_norm(self.vector)


@jax.jit
def _normalize(v):
    return v / jnp.linalg.norm(v)


@jax.jit
def _compute_norm(v):
    return (v.conj() * v).sum()
