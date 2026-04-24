# Transport

The transport is the single low-level thing TinyROS owns: a small
RPC-style pub/sub wire between nodes on one machine.

## Wire protocol

Every frame on the wire is prefixed with a fixed header:

```
+--------+-----------+------------------+
| 1 byte | 4 bytes   | N bytes          |
| kind   | length    | body             |
+--------+-----------+------------------+
```

Kinds:

| Kind | Meaning |
|---|---|
| `CALL` | RPC call: body is pickled `(req_id, cb_name, arg)` |
| `CALL_LARGE` | RPC call whose single-ndarray payload travels through shared memory; body is a pickled metadata dict (see below) |
| `REPLY` | Response to an RPC: body is pickled `(req_id, ok, payload_or_exc)` |
| `BYE` | Client announces graceful disconnect |

`arg` is a single positional value — the transport does not carry
`*args` / `**kwargs`. Callbacks bound via `TinyServer.bind` are invoked
as `fn(arg)`.

The `CALL` body uses **pickle protocol 5 with out-of-band buffers** so
numpy payloads embedded inside `arg` do not get copied into the main
pickle stream. The `CALL_LARGE` body is a plain pickle of the metadata
dict `{"req_id", "cb_name", "shm_name", "dtype", "shape", "nbytes"}`;
the ndarray itself travels through the shared-memory block named by
`shm_name`.

## Shared-memory fast path

Small messages travel inline on the socket. Messages whose *single-ndarray*
payload exceeds `TINYROS_SHM_THRESHOLD` bytes (default 64 KiB) take a
zero-copy side-channel:

1. Client allocates a `multiprocessing.shared_memory.SharedMemory` block.
2. Client copies the ndarray into it and sends only metadata on the socket.
3. Server maps the same block, copies the view out, unlinks.

The client tracks outstanding block names so segments never leak if the
send queue is dropped during shutdown.

## Endpoint abstraction

The transport uses TCP stream sockets (`AF_INET` / `SOCK_STREAM`). Each
node binds its own port from the network config; peers reach it via the
same `(host, port)` pair.

## Security posture

The wire deserializes with `pickle.loads`, which is equivalent to
arbitrary code execution for anyone who can connect to the port. There
is no authentication between peers. Consequences:

- **`TinyNode` defaults `bind_host` to `127.0.0.1`.** A node that
  forgets to pass `bind_host` is only reachable from the same host.
- Passing a non-loopback `bind_host` (`0.0.0.0`, a specific NIC, or a
  hostname) is allowed but logs a warning: the resulting node will
  execute arbitrary Python for any peer on the network that can
  reach the port.
- TinyROS is designed for single-host multi-process deployments
  behind a trust boundary (one user, one machine, one container).
  Running it across an untrusted network requires an external layer
  — mTLS, a VPN, IP allowlisting — that TinyROS does not provide.

## Threading model

Each server runs:

- An **accept** thread (polls with a 100 ms timeout so shutdown can
  break out).
- One **reader** thread per connected client, which dispatches
  CALL / CALL_LARGE / BYE frames. Peer sockets carry a 1 s read timeout
  so a silent peer cannot block shutdown.
- A bounded **thread pool** for callback execution so slow handlers do
  not block subsequent frames on the same connection.

Each client runs:

- A **send** thread that drains a queue of outbound frames.
- A **recv** thread that demultiplexes REPLY frames onto per-request
  futures. The recv socket uses the same 1 s read timeout pattern for
  responsive shutdown.

## Shutdown semantics

- Server shutdown closes the listen socket, shuts down every live peer
  socket to break reader `recv()` calls, drains the callback queue, then
  joins every thread with a caller-provided timeout.
- Client shutdown enqueues a `BYE` frame, then the stop sentinel, then
  reaps any in-flight shm blocks that never made it to the wire.

## Environment variables

| Name | Meaning | Default |
|---|---|---|
| `TINYROS_SHM_THRESHOLD` | Payload size (bytes) above which ndarrays go via shm | `65536` |
| `TINYROS_MAX_FRAME_BYTES` | Upper bound on inline frame body size; oversized headers are rejected without buffering | `268435456` (256 MiB) |

Set `TINYROS_SHM_THRESHOLD=0` to force everything inline.

Both limits can also be overridden per instance via the `shm_threshold`
and `max_frame_bytes` kwargs on `TinyServer` / `TinyClient`. The frame
cap only affects inline `CALL` and `REPLY` bodies; ndarrays that take
the shared-memory side-channel send only small metadata on the socket
and are unaffected.
