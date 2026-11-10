"""Versioned, finite research templates and local dataset admission checks.

Template priority and method versions are declared before observing their output.
No network acquisition occurs during planning, status reads or execution.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from gig.research_org.treasury_experiment import FIXTURE, PROVENANCE
from gig.research_org.treasury_experiment import HYPOTHESIS_ID as TREASURY_PARENT


@dataclass(frozen=True)
class ResearchTemplate:
    objective_id: str
    hypothesis_id: str
    dataset_id: str
    job_id: str
    runner: str
    question: str
    method: str
    method_version: int
    priority: int
    seq: int
    parent: str | None = None

    def job(self) -> dict[str, Any]:
        return {"seq": self.seq, "job_id": self.job_id,
                "hypothesis_id": self.hypothesis_id, "runner": self.runner}


TEMPLATES = (
    ResearchTemplate(
        "treasury-reversal-robustness-v1", "treasury-10y-reversal-robustness-2023-v1",
        "us-treasury-daily-par-yield-2023", "job-treasury-reversal-robustness-v1", "treasury_reversal",
        "Challenge the negative Treasury persistence finding with purged reversal robustness.",
        "purged-reversal-training-majority", 1, 100, 3, TREASURY_PARENT),
    ResearchTemplate(
        "ecb-fx-correlation-stability-v1", "ecb-fx-correlation-stability-2023-v1",
        "ecb-reference-fx-2023-usd-gbp-jpy", "job-ecb-fx-correlation-stability-v1", "ecb_risk",
        "Does training-window FX covariance remain calibrated on later ECB reference observations?",
        "chronological-covariance-stability", 1, 90, 4),
    ResearchTemplate(
        "boc-usdcad-variance-calibration-v1", "boc-usdcad-variance-calibration-2023-v1",
        "boc-valet-fxusdcad-2023", "job-boc-usdcad-variance-calibration-v1", "boc_volatility",
        "Does a causal variance forecast improve on training-only constant variance for Bank of Canada reference observations?",
        "purged-ewma-qlike-calibration", 1, 80, 5),
)


def find_template(objective_id: str) -> ResearchTemplate:
    match = next((row for row in TEMPLATES if row.objective_id == objective_id), None)
    if match is None:
        raise ValueError("unregistered research objective")
    return match


def dataset_readiness(dataset_id: str) -> dict[str, Any]:
    """Detached, bounded local provenance/schema/hash validation, never an experiment."""
    if dataset_id == TEMPLATES[0].dataset_id:
        try:
            meta = json.loads(PROVENANCE.read_text(encoding="utf-8"))
            digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
            if meta["dataset_id"] != dataset_id or digest != meta["sha256"]:
                raise ValueError("source identity")
            return {"dataset_id": dataset_id, "state": "AVAILABLE", "reason": "Pinned official Treasury source hash verified.",
                    "sha256": digest, "provider": "U.S. Treasury", "source_url": meta["source_url"],
                    "license_url": meta["license"], "acquired_at": meta["retrieved_at"],
                    "publication_time": "UNKNOWN", "revision_knowledge_time": "UNKNOWN"}
        except (OSError, ValueError, KeyError, TypeError):
            return {"dataset_id": dataset_id, "state": "ERROR", "reason": "Pinned Treasury evidence is unavailable or invalid.", "sha256": None}
    if dataset_id not in {row.dataset_id for row in TEMPLATES}:
        return {"dataset_id": dataset_id, "state": "UNAVAILABLE", "reason": "Dataset is not registered.", "sha256": None}
    try:
        from gig.research_org.observed_experiments import dataset_readiness as observed_readiness
        result = observed_readiness(dataset_id)
        # Normalize to detached JSON and deny incomplete identity/readiness claims.
        result = json.loads(json.dumps(result, allow_nan=False))
        if (not isinstance(result, dict) or result.get("dataset_id") != dataset_id
                or result.get("state") not in {"AVAILABLE", "UNAVAILABLE", "ERROR"}):
            raise ValueError("dataset readiness contract")
        source_digest = result.get("sha256")
        if result["state"] == "AVAILABLE" and (not isinstance(source_digest, str) or len(source_digest) != 64
                                                 or any(c not in "0123456789abcdef" for c in source_digest)):
            raise ValueError("dataset source hash")
        return result
    except ImportError:
        return {"dataset_id": dataset_id, "state": "UNAVAILABLE", "reason": "Approved observed-data runtime is unavailable.", "sha256": None}
    except (OSError, ValueError, TypeError, KeyError):
        return {"dataset_id": dataset_id, "state": "ERROR", "reason": "Pinned observed-data evidence failed admission checks.", "sha256": None}


def selection_identity(template: ResearchTemplate, dataset: dict[str, Any]) -> str | None:
    if dataset.get("state") != "AVAILABLE":
        return None
    raw = json.dumps({"objective_id": template.objective_id, "hypothesis_id": template.hypothesis_id,
                      "dataset_id": template.dataset_id, "dataset_sha256": dataset["sha256"],
                      "method": template.method, "method_version": template.method_version}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def run_ecb_risk():
    from gig.research_org.observed_experiments import run_ecb_risk as run
    return run()


def run_boc_volatility():
    from gig.research_org.observed_experiments import run_boc_volatility as run
    return run()
