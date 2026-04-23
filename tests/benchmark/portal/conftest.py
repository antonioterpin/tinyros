"""Fixtures for portal tests."""

from collections.abc import Callable

import jax
import numpy as np
import pytest


@pytest.fixture(params=["cpu", "gpu"])
def payload_factory(request: pytest.FixtureRequest) -> Callable:
    """Factory to create payloads on CPU or GPU."""
    pub_hw = request.param

    def make_payload(shape: tuple[int, int]) -> dict[str, object]:
        """Create a payload of given shape on specified hardware."""
        arr = np.zeros(shape, dtype=np.float32)

        if pub_hw == "cpu":
            payload = arr
        else:
            try:
                devices = jax.devices("gpu")
            except RuntimeError:
                pytest.skip("JAX GPU backend not available")

            if not devices:
                pytest.skip("No GPU devices visible to JAX")

            payload = jax.device_put(arr, devices[0])
            jax.block_until_ready(payload)

        return {
            "payload": payload,
            "pub_hw": pub_hw,
            "shape": shape,
            "bytes": arr.nbytes,
        }

    return make_payload
