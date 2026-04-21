"""Tests for the transport framing helpers.

Verifies that _pack_oob / _unpack_oob round-trip faithfully for scalars,
strings, and numpy arrays whose buffers travel out-of-band via pickle
protocol 5.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinyros.transport import _pack_oob, _unpack_oob


@pytest.mark.parametrize(
    "payload",
    [
        0,
        -1,
        3.14,
        "hello",
        b"raw bytes",
        (1, 2, 3),
        {"a": 1, "b": [1, 2]},
        None,
    ],
)
def test_small_scalars_roundtrip(payload: object) -> None:
    """Scalars and containers survive encode/decode unchanged."""
    decoded = _unpack_oob(_pack_oob(payload))
    assert decoded == payload, (
        f"expected {payload!r} after roundtrip, got {decoded!r}"
    )


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1,), np.float32),
        ((16, 16), np.float32),
        ((3, 4, 5), np.int64),
        ((0,), np.uint8),
    ],
)
def test_ndarray_roundtrip(
    shape: tuple[int, ...], dtype: type[np.generic]
) -> None:
    """ndarrays survive the oob path with shape, dtype, and content intact."""
    rng = np.random.default_rng(0)
    arr = rng.integers(-5, 5, size=shape).astype(dtype)
    decoded = _unpack_oob(_pack_oob(arr))
    assert isinstance(
        decoded, np.ndarray
    ), f"expected ndarray after roundtrip, got {type(decoded).__name__}"
    assert decoded.shape == shape, (
        f"shape mismatch: expected {shape}, got {decoded.shape}"
    )
    assert decoded.dtype == np.dtype(dtype), (
        f"dtype mismatch: expected {np.dtype(dtype)}, got {decoded.dtype}"
    )
    assert np.array_equal(
        decoded, arr
    ), "byte-identical payload expected after roundtrip"


def test_nested_ndarray_roundtrip() -> None:
    """ndarrays inside a tuple with other Python objects still roundtrip."""
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    encoded = _pack_oob((42, "label", arr))
    decoded = _unpack_oob(encoded)
    assert isinstance(
        decoded, tuple
    ), f"expected a tuple wrapper, got {type(decoded).__name__}"
    assert decoded[0] == 42, f"expected 42, got {decoded[0]!r}"
    assert decoded[1] == "label", f"expected 'label', got {decoded[1]!r}"
    assert np.array_equal(decoded[2], arr), "embedded ndarray lost content"
