# Sprint 8 bounded operational recovery

The active source checkout is `/tmp/gig-sprint7-publication`, descended from the
verified `development/gig-ui-redesign` publication. Its independent filesystem
has writable capacity. The original `/workspace` checkout, evidence, recovery
archives and Docker resources remain preserved.

## Current admission result

Read-only managed-local Docker observations identify the VFS driver at
`/var/lib/docker` and amd64 image
`sha256:c5446b87672bb72ee3fcf6809357c43dee2d9098d969a77c9c119ae1ab82325f`,
with reported size 697,300,556 bytes. Docker storage has **zero bytes available
to the caller**. Approximately 55 MB of reserved free blocks is not spendable
capacity. The current source and scratch directories share another filesystem,
which had approximately 2.4 GB available at observation time. That capacity
cannot fund VFS writes on the Docker-store filesystem.

No new container creation, startup, restart or image build was attempted in this
workstream. No image, cache, volume, source or evidence was deleted. Physical
ARM64/Ampere runtime acceptance remains unverified. Prior QEMU evidence and older
image results remain limited to their original source and emulated environment.

## Read-only capacity gate

```bash
python scripts/sprint8_operations_check.py \
  --source /tmp/gig-sprint7-publication \
  --scratch /tmp \
  --output /tmp/new-capacity-evidence.json
```

Exit zero means only that the conservative capacity plan was admitted. Exit two
means blocked or unknown. The script contains only Docker `info` and `image
inspect` calls against the managed local socket. It clears endpoint/context/TLS
selectors for those commands while preserving `DOCKER_CONFIG`, registry
configuration and inherited proxy settings. It never prints credentials, raw
exception text or the process environment. Output files must be new.

For a container, the Docker-store budget is the greater of 3 GiB and twice the
image size plus 512 MiB allocation allowance and 1 GiB reserve under VFS. Overlay
drivers use a smaller copy estimate but retain the same 3 GiB floor. Source work
reserves 128 MiB and scratch work 256 MiB. Reservations on the same device are
summed; inconsistent observations use the smaller available value. Unknown
driver, image, device or available-byte observations never admit an operation.

Build admission additionally requires a caller-declared bounded build-write
budget and at least 6 GiB in the Docker store. This is a conservative planning
gate, not proof that an unbounded build or concurrent writer cannot exhaust
storage. A separate future runner must enforce the recorded limits and observe
capacity again immediately before execution. Read-only container roots do not
avoid VFS image-copy allocation.

The actual Sprint 8 probe returned UNAVAILABLE with
`INSUFFICIENT_AVAILABLE_BYTES:docker`: 0 bytes available against a 3,221,225,472
byte store budget. Source/scratch passed their shared 402,653,184 byte budget.
Fresh image, container restart, container health and container crash-consistency
acceptance remain blocked by that precise capacity limitation. Storage recovery
must preserve protected work and unowned Docker resources; this tool does not
prune them automatically.

## Verification scope

Ten new regression checks cover the reproduced full-VFS/healthy-scratch failure,
same-device aggregation, exact byte boundaries, probe drift, unknown evidence,
build-write bounds, reserved blocks, local socket selection, sanitized failures
and rejection of credential-bearing or flag-like image inputs. Ruff, Pyflakes
and compilation pass for the owned script/tests.

A separate **Sprint 7 baseline installed wheel** check imported the actual x86
C++ extension from `/tmp/gig-sprint7-evaluation-installed`, verified its recorded
SHA-256 and passed 17 native/maintenance tests with five tests deliberately
deselected. An authored SQLite fixture was quiesced, backed up, restored to a new
directory and verified by exact file bytes and a read-only query. This checks
existing tooling and dependency availability; it is not final frozen Sprint 8
wheel or crash-consistency acceptance. Release-wide native/wheel checks are
coordinated separately after source freeze and must not be double-counted.

Safe metadata and exact scoped test evidence are under
`docs/evidence/sprint8/operations-*`; raw local outputs remain in
`/tmp/gig-sprint8-evidence/operations/`. These are local evidence files until
their publication and independent recovery are verified.
