# Local container verification

A CPU image was built from source using a multistage C++20 builder and a non-root runtime. Session proxy trust was provided as an optional BuildKit CA secret. Host compiled modules, credentials, caches, datasets, and model weights were excluded from the sanitized build context.

The recorded amd64 image ran with UID10001, a read-only root, network mode `none`, and no published ports. Actual HTTP health, admitted synthetic bars, and persisted market reads passed. After restart, GET-only checks verified identical source receipt and market content hashes without reingestion. No credentials were forwarded; no brokerage, public, or production deployment occurred. Only task-owned containers and volumes were cleaned.

See evidence/local-operations-runtime.json and evidence/local-operations-source.json for the tested image and source hashes. That image preceded the final workspace boundary/model/mutation repairs; its restart result must not be presented as a test of those later changes. The final installed native wheel separately passed all four HTTP workspace writes and nine persisted local views. A final cached image rebuild is the next operational verification. ARM64, systemd execution, browser accessibility, and production acceptance remain unverified.

## Exact published integration recheck

The source of commit `804e96fb34270c905a63b3081370ea2574223b0f` was independently materialized from Git objects and rebuilt with no host compiled modules or credentials. The new image passed the same non-root, read-only-root, network-none HTTP admission and GET-only restart checks with identical receipt and market content hashes. See evidence/container-804-runtime.json and container-804-source.json. Earlier disk exhaustion was resolved by removing only exact task-owned disposable images/cache records; no global prune or historical artifact change was used. This verifies the complete published workspace/native batch. Later optional market-foundation APIs are validated separately on the host and wheel, not retroactively claimed as container-tested.
