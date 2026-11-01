# ARM64 cross-compilation and emulation evidence

This cycle verifies a standalone C++20 numerical kernel on **emulated aarch64**.
It does not establish Ubuntu 24.04, Oracle Ampere A1, physical ARM64, full Python
extension, or ARM64 Docker runtime acceptance.

## What executed

The host was x86_64. Official Debian trixie GCC 14.2 cross-toolchain packages and
QEMU user 10.0.13 were downloaded in the earlier interrupted run, retained, and
verified against the package SHA-256 values in its authenticated APT index.
The source configuration uses the official Debian archive keyring. Twenty-five
packages were extracted privately beneath `/tmp/gig-arm64`; no package was
installed into the host system. Only the aarch64 QEMU executable was retained
from its multi-emulator package. Package hashes are recorded separately in
`arm64-toolchain-provenance.json` in the task evidence directory.

The real cross compiler built `src/cpp/rolling.cpp` and
`deploy/arm64/rolling_reference.cpp` with `-std=c++20 -O2 -static`. The fixture
has a compile-time `__aarch64__` guard. The runner independently read the output
ELF header and required **EM_AARCH64**, then executed the binary with explicit
QEMU user mode. No privileged container, kernel binfmt registration, daemon
restart, host networking, or Docker storage change was used.

The execution passed 3,966 comparisons against an independent two-pass scalar
reference, over windows 1, 2, 7, 64, and 252 and both supported variance degrees
of freedom. Maximum absolute mean difference was
`8.881784197001252e-16`; maximum standard-deviation difference was zero.
Four invalid-input cases also passed. This is correctness evidence, not a
performance benchmark: emulation timing does not predict Ampere performance.

## Reproduction

With a normal installed cross toolchain and QEMU user executable:

```sh
python scripts/check_arm64.py \
  --compiler aarch64-linux-gnu-g++ \
  --emulator qemu-aarch64 \
  --output work/arm64-execution.json
```

For this privately relocated toolchain, the actual invocation also supplied:

```text
--compiler /tmp/gig-arm64/root/usr/bin/aarch64-linux-gnu-g++
--emulator /tmp/gig-arm64/root/usr/bin/qemu-aarch64
--compiler-flag=--sysroot=/tmp/gig-arm64/root
--tool-library-dir /tmp/gig-arm64/root/usr/lib/x86_64-linux-gnu
```

The explicit library directory resolves relocated binutils libraries. It is
passed only to sanitized tool subprocess environments; brokerage credentials
and the parent process environment are not forwarded. Missing tools report
`BLOCKED`, never a successful native or hardware result. ELF parsing reads only
the fixed identification header. Four focused pytest tests cover evidence
classification, configuration, ELF identity, and environment admission.

## Python and deployment scope

Official PyPI version metadata was checked for 17 primary packages on
CPython 3.11 and 3.12 with aarch64 manylinux ABI requirements no newer than
glibc 2.39, the Ubuntu 24.04 baseline. Compatible binary wheels were listed for
all 34 package/version targets. This check uses tested host versions except
yfinance, which was not host-installed and was checked at its latest published
version. The report retains wheel filenames, source URLs, SHA-256 values, and
Python-version constraints. It **does not resolve the complete dependency
closure, install those wheels on ARM64, or execute ARM64 Python**. Torch,
Transformers, and restricted model weights are outside this wheel matrix.

Compose policy inspection passed loopback-only port binding, disabled broker
connectivity, persistent volumes, a read-only root, dropped capabilities,
no-new-privileges, memory limits, restart policy, and shutdown grace period.
These are static configuration checks, not ARM64 Compose execution. Existing
x86_64 container health/restart evidence remains separate.

Full ARM64 Python bindings, Docker image build/run, ARM64 health/restart/storage
behavior, physical Ampere execution, Ubuntu-specific systemd operation, and
hardware performance measurements remain **NOT_VALIDATED**. No account,
brokerage, paid VM, public deployment, or trading authority was involved.
