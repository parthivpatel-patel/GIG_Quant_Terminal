# Local container verification

A CPU image was built from source using a multistage C++20 builder and a non-root runtime. Session proxy trust was provided as an optional BuildKit CA secret. Host compiled modules, credentials, caches, datasets, and model weights were excluded from the sanitized build context.

The recorded amd64 image ran with UID10001, a read-only root, network mode `none`, and no published ports. Actual HTTP health, admitted synthetic bars, and persisted market reads passed. After restart, GET-only checks verified identical source receipt and market content hashes without reingestion. No credentials were forwarded; no brokerage, public, or production deployment occurred. Only task-owned containers and volumes were cleaned.

See evidence/local-operations-runtime.json and evidence/local-operations-source.json for the tested image and source hashes. That image preceded the final workspace boundary/model/mutation repairs; its restart result must not be presented as a test of those later changes. The final installed native wheel separately passed all four HTTP workspace writes and nine persisted local views. A final cached image rebuild is the next operational verification. ARM64, systemd execution, browser accessibility, and production acceptance remain unverified.
