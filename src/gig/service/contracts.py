"""Versioned research envelopes. Admission never confers broker authority."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gig.service.serialization import dumps, loads, normalize, public_error
from gig.timestamps import aware_timestamp


class State(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"
    STALE = "STALE"


class ProvenanceSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(min_length=1, max_length=1024)
    classification: Literal["REAL_TIME", "DELAYED", "SNAPSHOT", "HISTORICAL",
                            "SYNTHETIC", "MODEL_ESTIMATE", "UNAVAILABLE", "UNKNOWN"]
    as_of: str | None = None
    source_identity: str | None = None
    instrument_identity: str | None = None
    model_identity: str | None = None
    model_version: str | None = None
    evidence_state: str = "UNVERIFIED"
    readiness: str = "RESEARCH_ONLY"

    @field_validator("as_of")
    @classmethod
    def aware_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return aware_timestamp(value).isoformat()


class ErrorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str

    @model_validator(mode="after")
    def canonical_external_error(self) -> ErrorSchema:
        if self.model_dump() != public_error(self.code):
            raise ValueError("external errors require a supported public code and message")
        return self


class EnvelopeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1"] = "1"
    state: State
    data: Any
    provenance: ProvenanceSchema
    error: ErrorSchema | None = None
    scope: str = Field(min_length=1, max_length=256)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("data")
    @classmethod
    def serializable_data(cls, value: Any) -> Any:
        return normalize(value)

    @model_validator(mode="after")
    def semantic_contract(self) -> EnvelopeSchema:
        carries_data = self.state in (State.AVAILABLE, State.STALE)
        if carries_data != (self.data is not None):
            raise ValueError("state and data disagree")
        if self.state == State.ERROR and self.error is None:
            raise ValueError("error state requires a classified error")
        if self.state == State.AVAILABLE and self.error is not None:
            raise ValueError("available result cannot carry an error")
        if carries_data and self.provenance.classification in ("UNKNOWN", "UNAVAILABLE"):
            raise ValueError("data states require a known classification")
        if self.state == State.STALE and self.provenance.as_of is None:
            raise ValueError("stale data requires its original timestamp")
        body = self.model_dump(exclude={"content_sha256"})
        if hashlib.sha256(dumps(body).encode("utf-8")).hexdigest() != self.content_sha256:
            raise ValueError("contract content identity mismatch")
        return self


@dataclass(frozen=True, slots=True)
class Contract:
    """Immutable admitted bytes; each read is a fresh detached structure."""

    _wire: bytes

    def __post_init__(self) -> None:
        if not isinstance(self._wire, bytes):
            raise ValueError("contract wire must be immutable bytes")
        schema = EnvelopeSchema.model_validate(loads(self._wire))
        if dumps(schema.model_dump()).encode("utf-8") != self._wire:
            raise ValueError("contract wire must use canonical JSON encoding")

    @classmethod
    def admit(cls, state: State | str, data: Any, *, source: str,
              classification: str = "UNKNOWN", as_of: str | None = None,
              scope: str = "research", error: dict[str, str] | None = None,
              **provenance: Any) -> Contract:
        state = State(state)
        if state in (State.AVAILABLE, State.STALE) and data is None:
            raise ValueError("data required for available or stale result")
        if state in (State.UNAVAILABLE, State.UNKNOWN, State.ERROR) and data is not None:
            raise ValueError("non-data states must have null data")
        if state == State.ERROR and error is None:
            raise ValueError("error state requires a classified error")
        if state == State.AVAILABLE and error is not None:
            raise ValueError("available result cannot carry an error")
        if as_of is not None:
            as_of = aware_timestamp(as_of).isoformat()
        if state == State.STALE and as_of is None:
            raise ValueError("stale data requires its original timestamp")
        meta = ProvenanceSchema.model_validate({"source": source, "classification": classification,
                                                "as_of": as_of, **provenance})
        if state == State.AVAILABLE and classification in ("UNKNOWN", "UNAVAILABLE"):
            raise ValueError("available data requires a known classification")
        body = {"schema_version": "1", "state": state.value, "data": data,
                "provenance": meta.model_dump(), "error": error, "scope": scope}
        wire = dumps(body).encode("utf-8")
        body["content_sha256"] = hashlib.sha256(wire).hexdigest()
        EnvelopeSchema.model_validate(body)
        return cls(dumps(body).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return loads(self._wire)

    @property
    def wire(self) -> bytes:
        return self._wire
