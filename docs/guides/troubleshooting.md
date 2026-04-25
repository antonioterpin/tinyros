# Troubleshooting

Common operational failure modes when running TinyROS, and the
quickest path to diagnosis. Pair with
[transport-tunables.md](architecture/transport-tunables.md) when the
fix is a knob, and [transport.md](architecture/transport.md) when it
needs the wire-protocol context.

---

## A node will not shut down

**Symptoms.** `Ctrl-C` is sent to the example or your own program;
processes linger; subsequent runs fail with `OSError: [Errno 98]
Address already in use` (Linux) or `OSError: [WinError 10048]` (Windows).

**Likely causes.**

- A callback is stuck in a long-running operation. The `TinyServer`
  reader will block on the in-flight semaphore for up to
  `_INFLIGHT_ACQUIRE_POLL_S` between checks of the running flag, so
  shutdown is not instantaneous if the pool is saturated.
- A node was constructed but `shutdown()` was never called and the
  process exited via signal before the `atexit` hook ran.
- The example was killed with `SIGKILL` (`kill -9`); no Python-level
  cleanup runs, so server sockets and the receive thread can be left
  in `TIME_WAIT`.

**What to check.**

1. `lsof -i :<port>` (Linux/macOS) or `netstat -ano | findstr :<port>`
   (Windows) to confirm the port is held.
2. `ps -ef | grep python` to find lingering child processes from a
   `multiprocessing` parent that exited.

**Fix.**

- Kill the stragglers explicitly: `pkill -f tinyros` (Linux/macOS).
- Wait out `TIME_WAIT` (typically 60s) or pass a different port for
  the next run.
- For your own programs: always wrap node use in a `with TinyNode(...) as node:`
  block so cleanup runs deterministically.

---

## "Leaked" shared-memory segments

**Symptoms.** `/dev/shm/` (Linux) or `/tmp/` (macOS) accumulates
entries with names like `psm_<...>` after a crash. Disk fills up over
time across many crashed runs. On macOS you may see warnings of the
form `resource_tracker: There appear to be N leaked shared_memory
objects`.

**Why this happens.** The shared-memory side-channel is allocated by
the publisher, consumed by the subscriber, and unlinked by whichever
side last touched it. If the publisher process is killed mid-publish
(after `shm_open` but before the subscriber acks), the segment is
orphaned.

**What to check.**

- Linux: `ls -la /dev/shm/ | grep psm_`
- macOS: `ls -la /tmp/psm_*` (file paths vary by Python version)

**Fix.**

- Linux: `rm /dev/shm/psm_*` -- only safe when no TinyROS process is
  running, since you can unlink an in-use segment.
- macOS: same caveat; use `rm /tmp/psm_*`.
- Long term: prefer `with TinyNode(...) as ...:` and let the context
  manager handle teardown so `_try_unlink_shm` runs.

---

## Address already in use on restart

**Symptoms.** `OSError: [Errno 98] Address already in use` immediately
on launch, even though the previous process is gone.

**Why this happens.** The server enables `SO_REUSEADDR` on its listen
socket, but the OS still holds the port for clients in
`TIME_WAIT`. This is a TCP-level kernel state, not a TinyROS state.

**Fix.**

- Wait ~60s for `TIME_WAIT` to drain, or
- Change the port in `network_config.yaml` for the relaunch, or
- Confirm no other process is bound (`lsof -i :<port>`).

---

## Publishes silently succeed but subscribers receive nothing

**Likely causes** (in order of frequency):

1. The subscriber's callback name does not match the publish topic.
   Topics are routed by string match against bound callback names
   (`bind("scalar_data", ...)` <-> `publish("scalar_data", ...)`).
2. The peer is not listed in `network_config.yaml` -- TinyROS does
   not auto-discover.
3. The peer has not finished its connect handshake yet. First publish
   after process start may drop while clients are still reconnecting.
4. The reply path is failing silently because of a broad
   `except Exception` (see #44 typed-exception work for a long-term fix).

**What to check.**

- Set the goggles scope to verbose: `tinyros.transport` logs every
  inbound and outbound frame at DEBUG.
- Check `bind` was called before the first message landed.

---

## The example produces no log output

**Symptom.** `python main.py` runs but the screen stays blank apart
from process startup.

**Cause.** The example obtains its logger via
`tinyros.get_logger("tinyros.example", ...)`. Without the optional
`[logging]` extra (i.e. without `goggles` installed), it falls back to
the stdlib logger; the example calls `logging.basicConfig(...)` to
attach a console handler in that case. If `basicConfig` was removed
(or you are embedding the example without it), nothing has been
configured to print INFO messages.

**Fix.** Either install the `logging` extra
(`uv sync --extra logging`) or keep `logging.basicConfig` at process
entry.

---

## Pickle errors after upgrading peers asymmetrically

**Symptoms.** `pickle.UnpicklingError` or `AttributeError: Can't get
attribute ...` on inbound replies/calls.

**Cause.** Pickle exchanges custom Python objects. If two nodes
disagree on a class definition (different versions of the same
package, for example), the receiving side cannot reconstruct the
object.

**Fix.** Upgrade both peers in lockstep, or restrict the publish
payload to types both peers understand (primitives, ndarrays, dicts
of those).
