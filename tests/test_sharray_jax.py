import timeit
import time

import cupy as cp
import pytest


JAX_AVAILABLE = True
try:
    import jax
    import jax.numpy as jnp
    from tinyros.datatype.sharray.jax import jax_to_cupy, cupy_to_jax, JaxSharray

    jax.config.update("jax_enable_x64", True)
except ImportError:
    JAX_AVAILABLE = False

    class jnp:
        pass

    jnp.int32 = None
    jnp.float32 = None
    jnp.float64 = None

    def jax_to_cupy(x):
        raise ImportError("JAX is not available, cannot convert JAX to CuPy.")

    def cupy_to_jax(x):
        raise ImportError("JAX is not available, cannot convert CuPy to JAX.")


# NOTE: Most of the tests follow from the cupy tests.


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="JAX is not available, skipping JAX-related tests.",
)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((100_000_000,), jnp.int32),
        ((100_000_000,), jnp.float32),
        ((100_000_000,), jnp.float64),
    ],
)
def test_jax_to_cupy_timing(shape, dtype):
    jax_arr = jnp.arange(jnp.prod(jnp.array(shape)), dtype=dtype).reshape(shape)

    # warm-up
    _ = jax_to_cupy(jax_arr).data.ptr

    runs = 1_000
    threshold_ms = 0.006

    # measure total seconds for `runs` calls
    t_sec = timeit.timeit(lambda: jax_to_cupy(jax_arr).data.ptr, number=runs)

    avg_ms = (t_sec / runs) * 1_000
    assert (
        avg_ms < threshold_ms
    ), f"Average transfer time {avg_ms:.4f} ms exceeded {threshold_ms} ms"


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="JAX is not available, skipping JAX-related tests.",
)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((100_000_000,), jnp.int32),
        ((100_000_000,), jnp.float32),
        ((100_000_000,), jnp.float64),
    ],
)
def test_jax_to_cupy_no_dlpack_timing(shape, dtype):
    jax_arr = jnp.arange(jnp.prod(jnp.array(shape)), dtype=dtype).reshape(shape)

    # warm-up
    _ = jax_to_cupy(jax_arr).data.ptr

    runs = 1_000
    threshold_ms = 0.009

    t_sec = timeit.timeit(lambda: cp.array(jax_arr).data.ptr, number=runs)
    avg_ms = (t_sec / runs) * 1_000
    assert avg_ms > threshold_ms, (
        f"Average direct cp.array time {avg_ms:.4f} ms did not exceed "
        f"{threshold_ms} ms; you could skip the DLPack path."
    )


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="JAX is not available, skipping JAX-related tests.",
)
@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((100_000_000,), cp.int32),
        ((100_000_000,), cp.float32),
        ((100_000_000,), cp.float64),
    ],
)
def test_cupy_to_jax_no_dlpack_timing(shape, dtype):
    cp_arr = cp.arange(cp.prod(cp.array(shape)), dtype=dtype).reshape(shape)

    # warm-up
    _ = cupy_to_jax(cp_arr).block_until_ready()

    runs = 1_000
    threshold_ms = 0.055

    t_sec = timeit.timeit(lambda: cupy_to_jax(cp_arr).block_until_ready(), number=runs)
    avg_ms = (t_sec / runs) * 1_000
    assert (
        avg_ms < threshold_ms
    ), f"Average transfer time {avg_ms:.4f} ms exceeds {threshold_ms} ms"


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="JAX is not available, skipping JAX-related tests.",
)
def test_jax_copy_to_correctness():
    """Test that JaxSharray.copy_to correctly copies data between CuPy arrays."""
    a = jnp.arange(100, dtype=cp.int32)
    b = jnp.zeros_like(a)
    JaxSharray.copy_to(a, b)
    assert jnp.array_equal(a, b)

    a2 = jnp.arange(64, dtype=cp.float32).reshape(8, 8)
    b2 = jnp.zeros_like(a2)
    JaxSharray.copy_to(b2, a2)
    assert jnp.array_equal(a2, b2)


@pytest.mark.skipif(
    not JAX_AVAILABLE,
    reason="JAX is not available, skipping JAX-related tests.",
)
@pytest.mark.parametrize(
    "dim,threshold",
    [
        (100_000, 0.015),
        (1_000_000, 0.015),
        (10_000_000, 0.04),
        (100_000_000, 0.5),
    ],
)
def test_jax_copy_to_speed(dim, threshold):
    a = jnp.arange(dim, dtype=cp.float32)
    b = jnp.zeros_like(a)
    start = time.perf_counter()
    for _ in range(1000):
        JaxSharray.copy_to(a, b)
    end = time.perf_counter()
    elapsed = end - start  # ms average
    assert elapsed < threshold, (
        f"JaxSharray.copy_to took too long: {elapsed:.6f} ms on average."
        f"Threshold is {threshold} ms."
    )
