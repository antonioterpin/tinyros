# Transport

The transport is the single low-level thing TinyROS owns: a small
RPC-style pub/sub wire between nodes on one machine. It replaces the
former `portal`-based backend and uses the same techniques as
[`goggles`](https://github.com/antonioterpin/goggles)' `LocalTransport`.

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
| `CALL` | RPC call: body is pickled `(cb_name, args_tuple, req_id)` |
| `CALL_LARGE` | RPC call whose payload went through shared memory |
| `REPLY` | Response to an RPC: body is pickled `(req_id, ok, payload_or_exc)` |
| `BYE` | Client announces graceful disconnect |

The body uses **pickle protocol 5 with out-of-band buffers** so numpy
payloads do not get copied into the main pickle stream.

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

- **Linux / macOS**: `AF_UNIX` stream socket. Addresses are
  `(host, port)`; the concrete filesystem path is derived from these and
  protected at `0o600`.
- **Windows**: `AF_UNIX` is unreliable, so the transport binds a TCP
  loopback socket on `127.0.0.1` at the configured port.

Per-node addressing is inherited from the network config: each node binds
its own port, identical to the pre-portal behavior.

## Threading model

Each server runs:

- An **accept** thread (polls with a 100 ms timeout so shutdown can break out).
- One **reader** thread per connected client, which dispatches CALL / BYE frames.
- A **drain** thread that serializes callback invocations.

Each client runs:

- A **send** thread that drains a queue of outbound frames.
- A **recv** thread that demultiplexes REPLY frames onto per-request Futures.

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

Set `TINYROS_SHM_THRESHOLD=0` to force everything inline.
