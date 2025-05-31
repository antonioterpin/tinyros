import timeit

import jax.numpy as jnp
import numpy as np
import pytest

from tinyros.datatype import JaxDataType, SerializedCuPyArray, SerializedJaxArray


# ------------------------------------------------------------------------------
# Fixture for a large JAX array
# ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def large_jax():
    # use a NumPy buffer so the test doesn’t depend on device‐specific JAX RNG
    arr = np.random.rand(100_000_000).astype(np.float32)
    return jnp.array(arr)


# ------------------------------------------------------------------------------
# Low‐level unit tests
# ------------------------------------------------------------------------------
def test_can_serialize_and_deserialize():
    dt = JaxDataType()
    x = jnp.ones((10, 10))
    # can_serialize should pick up JAX ndarrays
    assert dt.can_serialize(x)
    ser = dt.serialize_data(x)
    assert isinstance(ser, SerializedJaxArray)
    # underlying payload should be a SerializedCuPyArray
    assert isinstance(ser.cupy_serialized, SerializedCuPyArray)

    # and we can round‐trip via the low‐level API
    assert dt.can_deserialize(ser)
    y = dt.deserialize_data(ser)
    np.testing.assert_array_equal(np.array(y), np.array(x))


# ------------------------------------------------------------------------------
# Round‐trip correctness
# ------------------------------------------------------------------------------
def test_jax_roundtrip_correctness(large_jax):
    obj = JaxDataType()
    obj.arr_jax = large_jax

    ser = obj.serialize()
    new_obj = JaxDataType().deserialize(ser)

    # shapes and dtypes should be preserved
    assert new_obj.arr_jax.shape == large_jax.shape
    assert new_obj.arr_jax.dtype == large_jax.dtype

    # values should round‐trip exactly
    np.testing.assert_array_equal(np.array(new_obj.arr_jax), np.array(large_jax))


# ------------------------------------------------------------------------------
# Performance tests
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "attr_name, data_fixture, threshold",
    [
        ("arr_jax", "large_jax", 2.1e-4),
    ],
)
def test_serialize_speed_jax(request, attr_name, data_fixture, threshold):
    arr = request.getfixturevalue(data_fixture)
    obj = JaxDataType()
    setattr(obj, attr_name, arr)

    # measure serialize()
    N = 100
    t = timeit.timeit(lambda: obj.serialize(), number=N)
    avg = t / N
    assert avg < threshold, f"serialize() too slow: {avg:.3f}s > {threshold}s"


@pytest.mark.parametrize(
    "attr_name, data_fixture, threshold",
    [
        ("arr_jax", "large_jax", 8e-5),
    ],
)
def test_deserialize_speed_jax(request, attr_name, data_fixture, threshold):
    arr = request.getfixturevalue(data_fixture)
    obj = JaxDataType()
    setattr(obj, attr_name, arr)
    ser = obj.serialize()

    # measure deserialize()
    N = 1000
    t = timeit.timeit(lambda: JaxDataType().deserialize(ser), number=N)
    avg = t / N
    assert avg < threshold, f"deserialize() too slow: {avg:.3f}s > {threshold}s"
