# Tiny objects: runtime behavior

Operational reference for `TinyServer`, `TinyClient`, and `TinyNode`. The wire protocol and framing live in [`transport.md`](transport.md); this page covers state machines, failure handling, and cross-process choreography that would otherwise bloat docstrings.

`help(...)` on each class still gives a concise summary; deeper rationale lives here.

---

## TinyServer

### Lifecycle

1. **`__init__`** — allocate the worker pool, the in-flight semaphore, and per-connection bookkeeping. No socket activity yet.
2. **`bind(name, fn)`** — register a callback. Must run before `start()`; the dispatch path reads `_callbacks` without locking, so accepting new bindings while inbound calls are in flight would race. `bind` after `start` raises `RuntimeError`.
3. **`start(block=False)`** — bind the listen socket, spawn the accept thread, and (optionally) join it. Synchronous: by the time `start` returns, peers can connect.
4. **`close(timeout=...)`** — clear the running flag, close the listen socket, and join the accept thread *first*. Only after the accept loop is gone do we snapshot and shut down every peer socket to break reader `recv()` calls (this avoids a race where a just-accepted connection could miss the snapshot and leak its reader thread); then join the reader threads with the caller's timeout. The worker pool is drained with `wait=True` afterwards, so an already-running callback can extend `close()` past the configured timeout. `cancel_futures=True` prevents queued-but-not-yet-running callbacks from starting.

### Worker pool and in-flight cap

Two limits gate dispatch:

- **`workers`** (default 32): the `ThreadPoolExecutor`'s max concurrent callbacks.
- **`max_in_flight`** (default `workers * 3`): the count of *running plus queued* callbacks before the reader thread is blocked from submitting more.

The reader acquires a `BoundedSemaphore` slot before every `pool.submit`; the done-callback releases it when the call completes. When the pool is saturated, the reader stalls on the semaphore, which stalls `recv` on the connection, which (via TCP flow control) pushes the queue growth back to the client's `sendall`. **No call is dropped — the client just slows down to match the server.**

`max_in_flight=0` (or any computed value `< 1`) is rejected at construction with `ValueError`; otherwise the reader would spin on timeouts and never dispatch.

### CALL_LARGE failure handling

`_execute_large_call` runs on the worker pool (not the reader thread, so a slow shm `memcpy` never blocks subsequent frames). Two failure modes:

- **Metadata pickle is corrupt.** No `req_id` to address a reply to, so we treat it as a protocol error and call `_drop_conn(conn)`. The client's recv loop notices the EOF and fails every pending future with `ConnectionError` in bounded time -- much better than letting the caller hang until some unrelated teardown lands.
- **Metadata parses but the shm block is missing or malformed.** Use the parsed `req_id` / `cb_name` to send a synthesized `ok=False` REPLY (`_reply_failure`). The caller's future resolves with a `RuntimeError` describing the materialization failure rather than hanging forever.

### Submit during shutdown

`ThreadPoolExecutor.submit` raises `RuntimeError` if the pool is already shutting down. `_submit_call` catches it, releases the semaphore slot, and returns quietly: the close path will tear the connection down on its own without a spurious "frame handler failed" log.

---

## TinyClient

### Lifecycle

1. **`__init__`** — call `_connect(connect_timeout)` (default 10 s), spawn the send thread, spawn the recv thread bound to the new socket. Returns when both threads are running.
2. **`call(method, arg)` / `client.method(arg)`** — assign a `req_id`, encode the frame (CALL or CALL_LARGE depending on argument shape and threshold), enqueue for the send thread, and return a `concurrent.futures.Future`.
3. **`close(timeout=...)`** — enqueue a `BYE` frame, then the stop sentinel, then reap any in-flight shm blocks that never made it to the wire.

### Reconnect on send failure

When `_send_loop`'s `sendall` raises `OSError`, the client does not tear down — it attempts a single best-effort reconnect.

The choreography (steps 2–4 run under `_pending_lock`; `call()` takes the same lock around its insert+enqueue, so a concurrent `call()` cannot interleave a frame past the failure handling):

1. **Log the failure** at `WARN` and build a `ConnectionError` to surface to callers.
2. **Fail every pending future** by snapshotting and clearing `_pending`. The frame in flight may or may not have been processed by the server; there is no safe layer-7 retry.
3. **Unlink the failed frame's shm block** (`_unlink_orphan_shm`). The server never consumed it, so unlink responsibility falls back on the client. No-op for inline frames.
4. **Drain the send queue** (`_drain_queued_sends`). Their futures have already been failed; transmitting them on a reconnected socket would only produce stale server-side side effects and orphan REPLY frames. Each drained CALL_LARGE has its shm block unlinked too. The stop sentinel is preserved so `close()` can still terminate the loop.
5. **Try to reconnect** (`_try_reconnect`) — but only if `_running` is still set and `_shutdown_called` is false. If another path already tore the client down (notably `_recv_loop`'s oversized-reply branch, which clears `_running`), skip reconnect and let the loop exit cleanly:
   - Shut down and close the old socket (wakes the old recv thread on EOF).
   - Briefly join the old recv thread.
   - Call `_connect(reconnect_timeout)` (default 2 s) in a retry loop with a small fixed delay (50 ms) between attempts.
   - On success: swap in the new socket, spawn a fresh recv thread bound to it. The send loop resumes; subsequent calls succeed against the new server.
   - On failure: the client is torn down permanently — `_stop_running` + `_shutdown_io`.

One reconnect per detected socket failure. Callers that need long-running resilience can either retry across the same client (the next send triggers another reconnect attempt) or recreate the client.

### shm bookkeeping across reconnect

Outstanding shm block names live in `_pending_shm`. Three release paths:

| Path | Release |
|---|---|
| Frame sent successfully | Server unlinks after consumption; client just discards the tracking entry. |
| Send-failure path / queue-drain path | `_unlink_orphan_shm` (the server will never see the block). |
| `close()` after failure | `_shutdown_io` reaps any stragglers still in the set. |

This guarantees `/dev/shm` does not accumulate orphan blocks across repeated send failures.

---

## TinyNode

### Lifecycle

1. **`__init__`**:
   1. Allocate the underlying `TinyServer` (no socket activity yet).
   2. **`_setup_subscriptions`** — bind every subscriber callback by name on the server.
   3. `atexit.register(self.shutdown)` — best-effort cleanup hook.
   4. **`server.start(block=False)`** — open the listen socket. The node is now reachable from peers.
   5. **`_setup_publishing`** — open one `TinyClient` per distinct `(host, port)` this node publishes to.
2. **`publish(topic, msg)`** — for each subscription on `topic`, dispatch the message via the corresponding client and return a list of futures.
3. **`shutdown()`** — close every client, close the server, unregister the atexit hook.

### Why bind-then-start-then-dial

The order matters: every `TinyClient.__init__` blocks for up to `connect_timeout` (10 s default) until the peer's listen socket is up. If the order were "open clients, then start server", a multi-process topology with cyclic publishes (A ↔ B) would deadlock — each node blocks dialing the other before either listen socket exists.

With the listen socket up before any outbound dialing, peers can connect to us as soon as their own setup reaches the publishing step. The connect-retry budget is for unrelated "peer process is still cold-starting" cases, not for resolving a self-inflicted deadlock.

### publish fan-out

`publish` looks up the topic in `topic_calls`, dispatches to every subscriber, and returns one future per subscriber. Topics with no subscribers warn and return an empty list — publishers are allowed to run before consumers connect.

A future failing here means a single subscriber's RPC failed (server down, callback raised, etc.). Other subscribers' futures resolve independently — there is no all-or-nothing semantic.

---

## Cross-cutting

### Shutdown timeouts

`TinyServer.close(timeout=...)` and `TinyClient.close(timeout=...)` both use `timeout` for thread joins, but they diverge after that:

- **Server**: the worker pool is drained with `wait=True` *after* the joins, so an already-running callback can extend `close()` past the budget. The timeout does not bound this drain.
- **Client**: the send/recv joins are themselves bounded by `timeout`. There is no equivalent unbounded drain. The only path that can block indefinitely is the `timeout=None` overload (which is what the parameter name implies on POSIX).

For a hard upper bound on either side, run the node in a supervisor and reap externally.

### Unknown frame kinds

Both reader loops log unexpected non-CALL/non-REPLY frame kinds at `WARN` rather than crashing. This makes protocol drift / version skew debuggable instead of silently dropping the frame.

### Frame-size cap

`_max_frame_bytes` (default 256 MiB, env `TINYROS_MAX_FRAME_BYTES`) caps inline `CALL` and `REPLY` body sizes. A peer claiming more in the header is torn down without buffering. The cap does not apply to `CALL_LARGE` bodies — those carry only metadata; the actual payload travels through shared memory.
