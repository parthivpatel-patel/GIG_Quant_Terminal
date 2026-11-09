# PR #10 worker CPU budget repair

GitHub CI run 41962999605 failed six worker tests on Python 3.12 at
`5fe1273c80e601dc799469a83312b2f076eb00ca`. The worker compared its
120-second allowance to the server process's entire lifetime CPU usage.
Earlier numerical tests could therefore exhaust the allowance before a worker
had started. The frontend job passed; the backend job reported 1,024 passed,
6 failed, 11 skipped and 597 passing subtests.

The allowance now measures conservative process-wide CPU elapsed since this
worker object's first explicit start. This includes concurrent server threads,
and retains the baseline across pause/resume or another start on the same
object. It is not a per-thread CPU meter or a durable cross-process quota.
The job, storage and memory limits remain enforced.

Two regression tests first reproduced the defect with 1,000 seconds of prior
process CPU. They verify a fresh worker completes its approved queue, the exact
budget boundary, subsequent exhaustion and preservation across another start.

Local verification on Python 3.12: worker and research organization suites,
16 passed; focused Ruff, Pyflakes and diff whitespace checks passed.
HTTP checks require the environment's permitted socket-capable execution scope;
the restricted sandbox also hangs a minimal AnyIO threadpool probe. No
application behavior was changed to work around that sandbox limitation.

PR #10 must still pass its GitHub checks before integration into the development
branch. This repair does not authorize brokerage access or expand the backlog.
