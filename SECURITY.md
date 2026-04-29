# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | yes       |
| < 0.4   | no        |

## Reporting a vulnerability

Please report security issues privately to **antonio.terpin@gmail.com**.

If the issue is in the RPC transport (`tinyros.transport`), include:

- a minimal reproduction (network config + sequence of calls);
- whether the trigger requires only a localhost peer or non-loopback
  network access;
- the tinyros version
  (`python -c "import tinyros; print(tinyros.__version__)"`).

The maintainer will acknowledge receipt within 7 days and aim to ship a
fix and a coordinated disclosure within 30 days for confirmed issues.
Public discussion happens after a fix is released.

## Threat model snapshot

- The RPC server uses `pickle.loads` on incoming frames, so any peer
  that can connect to a server has effective code-execution authority.
  Bind to loopback unless every peer is fully trusted; non-loopback
  binds emit a warning.
- The shared-memory fast path uses POSIX shm names with no
  authentication; the same trust model applies.

These limits are documented behavior of the current transport and are
not themselves vulnerabilities under this policy. Bypasses of those
limits (e.g. crashing the server from a peer that should be benign,
escaping the configured frame-size cap) are in scope.
