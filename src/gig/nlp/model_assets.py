"""Explicit, pinned FinBERT research-asset acquisition; inference never downloads.

The official checkpoint is CC-BY-NC-SA-4.0. This helper requires an explicit
noncommercial-research acknowledgement and records attribution and integrity.
It grants neither commercial use nor trading authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

MODEL_ID = "ProsusAI/finbert"
PINNED_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
LICENSE = "cc-by-nc-sa-4.0"
FILES = frozenset(
    {
        "README.md",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
)
MAX_ASSET_BYTES = 512 * 1024 * 1024


def _directory(path: str | Path) -> Path:
    destination = Path(path).absolute()
    if any(item.is_symlink() for item in (destination, *destination.parents)):
        raise ValueError("model asset directory cannot use symbolic links")
    return destination


def _fetch(url: str, *, timeout: float = 30):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "gig-local-noncommercial-finbert-research/1"}
        ),
        timeout=timeout,
    )


def prepare_finbert_assets(
    directory: str | Path,
    *,
    allow_noncommercial_research: bool = False,
    revision: str = PINNED_REVISION,
    opener=None,
) -> dict[str, Any]:
    """Download only official pinned safetensors/tokenizer assets, verify, publish.

    Network use is explicit here and nowhere in LocalFinBERT. Existing files are
    verified or rejected; they are never overwritten. ``opener`` exists for
    deterministic offline transport tests, not to accept arbitrary remote URLs.
    """
    if allow_noncommercial_research is not True:
        raise ValueError("CC-BY-NC-SA-4.0 noncommercial research acknowledgement required")
    if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise ValueError("an immutable 40-character source revision is required")
    root = _directory(directory)
    if root.exists() and not root.is_dir():
        raise ValueError("model asset destination must be a directory")
    transport = opener or _fetch
    metadata_url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{revision}?blobs=true"
    with transport(metadata_url) as response:
        metadata_bytes = response.read(1_048_577)
    if len(metadata_bytes) > 1_048_576:
        raise ValueError("model metadata exceeds declared bound")
    metadata = json.loads(metadata_bytes)
    if (
        metadata.get("id") != MODEL_ID
        or metadata.get("sha") != revision
        or metadata.get("cardData", {}).get("license") != LICENSE
        or metadata.get("gated")
    ):
        raise ValueError(
            "checkpoint identity, immutable revision, or permitted research license mismatch"
        )
    siblings = {
        item["rfilename"]: item
        for item in metadata.get("siblings", [])
        if isinstance(item, dict) and item.get("rfilename") in FILES
    }
    if set(siblings) != FILES:
        raise ValueError("official safetensors/tokenizer assets are incomplete")
    root.mkdir(parents=True, exist_ok=True)
    artifacts, digests = {}, {}
    for name in sorted(FILES):
        descriptor = siblings[name]
        size = descriptor.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_ASSET_BYTES:
            raise ValueError("asset size is absent or outside declared limits")
        lfs = descriptor.get("lfs")
        expected = lfs.get("sha256") if isinstance(lfs, dict) else descriptor.get("blobId")
        algorithm = "sha256" if lfs else "git-blob-sha1"
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[a-f0-9]{64}" if lfs else r"[a-f0-9]{40}", expected) is None
        ):
            raise ValueError("official asset integrity identity required")
        target = root / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError("asset path must be a regular nonsymlink file")
        source_url = f"https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}"
        sha256 = hashlib.sha256()
        source_digest = (
            hashlib.sha256() if lfs else hashlib.sha1(b"blob " + str(size).encode() + b"\0")
        )
        temporary = None
        try:
            if target.exists():
                stream = target.open("rb")
                output = None
            else:
                stream = transport(source_url)
                descriptor_fd, temporary = tempfile.mkstemp(prefix=".finbert-download-", dir=root)
                output = os.fdopen(descriptor_fd, "wb")
            count = 0
            try:
                with stream:
                    while chunk := stream.read(1_048_576):
                        count += len(chunk)
                        if count > size:
                            raise ValueError("asset response exceeds declared source size")
                        sha256.update(chunk)
                        source_digest.update(chunk)
                        if output is not None:
                            output.write(chunk)
                if output is not None:
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                if output is not None:
                    output.close()
            if count != size or source_digest.hexdigest() != expected:
                raise ValueError("official asset size or content digest mismatch")
            if temporary is not None:
                os.link(temporary, target)  # Never replace surviving assets.
            digests[name] = sha256.hexdigest()
            artifacts[name] = {
                "url": source_url,
                "bytes": size,
                "sha256": sha256.hexdigest(),
                "source_digest": expected,
                "source_digest_algorithm": algorithm,
            }
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
    manifest = {
        "model_id": MODEL_ID,
        "revision": revision,
        "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": "ProsusAI/finbert; Dogu Araci, FinBERT: Financial Sentiment Analysis with Pre-trained Language Models (2019)",
        "scope": "NONCOMMERCIAL_RESEARCH_ONLY",
        "trading_approved": False,
        "commercial_use_approved": False,
        "source_metadata_url": metadata_url,
        "source_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "files": digests,
        "artifacts": artifacts,
    }
    content = json.dumps(
        manifest, sort_keys=True, allow_nan=False, ensure_ascii=False, separators=(",", ":")
    ).encode()
    from gig.artifacts import write_immutable

    write_immutable(root / "finbert-provenance.json", content)
    return {
        "directory": str(root),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "model_id": MODEL_ID,
        "revision": revision,
        "license": LICENSE,
        "scope": "NONCOMMERCIAL_RESEARCH_ONLY",
        "assets_verified": True,
        "trading_approved": False,
        "commercial_use_approved": False,
    }
