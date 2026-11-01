# Recovery artifacts

The manifest identifies the exact tested source commit. The Git bundle contains complete Git history/source through that commit. The evidence archive contains the selected committed reports and checksums. These artifacts contain no runtime credentials and do not claim to back up a running brokerage account or unsaved operational state.

Verify both SHA-256 checksums before use, then run `git bundle verify checkpoint-e19d688.bundle` and clone it into a new directory. Do not restore over a surviving or protected workspace. The bundle covers the tested source commit; this later artifact-only commit cannot be contained in its own bundle.

The source branch is published independently to GitHub. External artifact durability is established only when its downloaded bytes match this manifest; the local manifest alone is not evidence of an external copy. No institutional, economic, broker, ARM64, or T4 certification is implied.

Independent verification has now completed: both artifacts were downloaded from immutable GitHub URLs, their SHA-256 checksums matched, and the downloaded bundle passed verification, a fresh no-checkout clone, and full Git fsck. See external-verification.json. This verifies the source checkpoint and selected evidence; it does not claim to preserve live account or external environment state.
