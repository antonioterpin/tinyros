# Transport tunables

Operator-facing reference for every knob TinyROS's transport layer
exposes -- environment variables that override defaults at runtime,
constructor arguments that override at instantiation, and the
in-source constants worth knowing about for future tuning work.

The ground truth lives in
[src/tinyros/transport/_common.py](../../../src/tinyros/transport/_common.py);
this document is a flattened view of it for readers who do not want to
grep the source.

---

## Environment variables

These are read once when the first client/server is constructed (per
process) and used as defaults for the corresponding constructor
arguments.

| Variable | Default | Effect |
|---|---|---|
| `TINYROS_SHM_THRESHOLD` | `65536` (64 KiB) | Minimum ndarray size in bytes that switches the publish to the shared-memory side-channel. Smaller arrays are pickled inline. Set to `0` to force shm for any ndarray; set to a very large number to disable shm. |
| `TINYROS_MAX_FRAME_BYTES` | `268435456` (256 MiB) | Upper bound the reader enforces on the `length` field of the inline wire frame. Frames claiming more are rejected without buffering, so a misbehaving peer cannot trigger an unbounded allocation. Does **not** affect ndarray-via-shm payloads -- only metadata travels inline for those. |

Both accept a non-negative integer. A non-integer value falls back to
the default with a logged warning.

---

## Constructor arguments

The values below override the env-derived defaults on a per-instance
basis. Pass them to `TinyClient(...)` / `TinyServer(...)` when
embedding -- env vars cover the typical case.

### `TinyClient`

| Argument | Default | Effect |
|---|---|---|
| `shm_threshold` | `TINYROS_SHM_THRESHOLD` or `65536` | Per-client shm cutoff. |
| `max_frame_bytes` | `TINYROS_MAX_FRAME_BYTES` or `268435456` | Per-client inline frame ceiling enforced on inbound replies. |

### `TinyServer`

| Argument | Default | Effect |
|---|---|---|
| `workers` | `32` | Maximum concurrent callback executions (thread-pool size). |
| `max_in_flight` | `workers * 3` | Maximum calls in flight (running + queued). Backpressure: readers block on this semaphore before scheduling a new call. |
| `max_frame_bytes` | `TINYROS_MAX_FRAME_BYTES` or `268435456` | Per-server inline frame ceiling enforced on inbound calls. |

---

## Compile-time constants

These live in [`_common.py`](../../../src/tinyros/transport/_common.py)
and are not env-overridable today. They influence shutdown latency and
loop tightness; documented here so a future tuning PR has the context.

| Constant | Default | Effect |
|---|---|---|
| `_LISTEN_BACKLOG` | `64` | Listen-queue depth for the server socket. |
| `_CONNECT_TIMEOUT_S` | `10.0` | Initial connect deadline for a client. |
| `_RECONNECT_TIMEOUT_S` | `2.0` | Reconnect attempt deadline after a peer drop. |
| `_ACCEPT_POLL_S` | `0.1` | Server accept-loop poll grain -- bounds shutdown latency. |
| `_READ_POLL_S` | `1.0` | Per-connection reader poll grain -- bounds shutdown latency. |
| `_INFLIGHT_ACQUIRE_POLL_S` | `0.2` | Reader's wait grain for an in-flight slot. Short enough that `close()` is responsive; long enough that a saturated pool does not burn CPU. |

---

## Tuning notes

- **Throughput on small ndarrays**: raise `TINYROS_SHM_THRESHOLD` to
  keep small arrays inline (avoids shm setup cost). The 64 KiB default
  already covers most scalar / control-loop traffic.
- **Throughput on large ndarrays**: lower the threshold (or set to
  `0`) to push more onto shm, which avoids serializing the buffer.
- **Memory ceiling per inbound frame**: lower `TINYROS_MAX_FRAME_BYTES`
  to bound the worst-case allocation a single frame can request.
- **Pool starvation**: long-running callbacks can saturate `workers`;
  additional callers block on `_INFLIGHT_ACQUIRE_POLL_S` boundaries
  until a slot frees. Pass a larger `workers` / `max_in_flight` to the
  `TinyServer` constructor if the workload demands it.
