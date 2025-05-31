import timeit

import cupy as cp
import numpy as np
import pytest

from tinyros.datatype import DataType


# ------------------------------------------------------------------------------
# Fixtures for large arrays
# ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def large_np():
    return np.random.rand(100_000_000)


@pytest.fixture(scope="module")
def large_cp():
    return cp.random.rand(100_000_000)


# ------------------------------------------------------------------------------
# Correctness tests
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "with_numpy, with_cupy",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_roundtrip_correctness(large_np, large_cp, with_numpy, with_cupy):
    obj = DataType()
    if with_numpy:
        obj.arr_np = large_np
    if with_cupy:
        obj.arr_cp = large_cp

    ser = obj.serialize()

    new_obj = DataType().deserialize(ser)

    if with_numpy:
        # exactly equal after roundtrip
        np.testing.assert_array_equal(new_obj.arr_np, large_np)

    if with_cupy:
        # same shape/dtype, and same values on GPU
        assert new_obj.arr_cp.shape == large_cp.shape
        assert new_obj.arr_cp.dtype == large_cp.dtype
        cp.testing.assert_allclose(new_obj.arr_cp, large_cp)


# ------------------------------------------------------------------------------
# Performance tests
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "attr_name, data_fixture, threshold",
    [
        ("arr_cp", "large_cp", 1.2e-5),
    ],
)
def test_serialize_speed(request, attr_name, data_fixture, threshold):
    """Ensure that serializing a single large array is faaaast."""
    arr = request.getfixturevalue(data_fixture)
    obj = DataType()
    setattr(obj, attr_name, arr)

    # measure serialize()
    N = 100
    t = timeit.timeit(lambda: obj.serialize(), number=N)
    avg = t / N
    assert avg < threshold, f"serialize() too slow: {avg:.3f}s > {threshold}ms"


@pytest.mark.parametrize(
    "attr_name, data_fixture, threshold",
    [
        ("arr_cp", "large_cp", 2e-5),
    ],
)
def test_deserialize_speed(request, attr_name, data_fixture, threshold):
    """Ensure that deserializing a single large is fast."""
    arr = request.getfixturevalue(data_fixture)
    obj = DataType()
    setattr(obj, attr_name, arr)
    ser = obj.serialize()

    # measure deserialize()
    N = 100
    t = timeit.timeit(lambda: DataType().deserialize(ser), number=N)
    avg = t / N
    assert avg < threshold, f"deserialize() too slow: {avg:.3f}s > {threshold}s"
