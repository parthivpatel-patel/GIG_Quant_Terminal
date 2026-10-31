"""Bounded, deterministic JSON boundary with explicit lossless scalar encodings.

Non-finite values are rejected. Decimal and integers outside JavaScript's exact
range use tagged objects; callers must not coerce those tags back into numbers.
Timestamp objects require a timezone and are emitted in UTC without truncation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd


class SerializationError(ValueError):
    """Public code only: never retain an untrusted object's repr or provider error."""

    def __init__(self, code: str = "INVALID_JSON") -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 1_048_576
    max_depth: int = 32
    max_nodes: int = 50_000
    max_items: int = 10_000
    max_string: int = 65_536

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_depth", "max_nodes", "max_items", "max_string"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("serialization limits must be nonnegative integers")
        if self.max_depth > 128:
            raise ValueError("serialization recursion limit cannot exceed 128")


DEFAULT_LIMITS = Limits()
MAX_EXACT_INTEGER = 2**53 - 1


def normalize(value: Any, *, limits: Limits = DEFAULT_LIMITS) -> Any:
    """Take a detached JSON snapshot. Cycles and unsupported objects fail closed."""
    nodes = 0
    active: set[int] = set()

    def visit(obj: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > limits.max_nodes:
            raise SerializationError("TOO_MANY_NODES")
        if depth > limits.max_depth:
            raise SerializationError("TOO_DEEP")
        if obj is None or isinstance(obj, (bool, np.bool_)):
            return None if obj is None else bool(obj)
        if isinstance(obj, str):
            if len(obj) > limits.max_string:
                raise SerializationError("STRING_TOO_LONG")
            try:
                obj.encode("utf-8", errors="strict")
            except UnicodeError:
                raise SerializationError("INVALID_UTF8") from None
            return obj
        if isinstance(obj, (int, np.integer)):
            integer = int(obj)
            if abs(integer) <= MAX_EXACT_INTEGER:
                return integer
            # Python's decimal-digit security cap may be lower than max_string.
            # Fail with a public code instead of leaking an interpreter error.
            try:
                text = str(integer)
            except ValueError:
                raise SerializationError("INTEGER_TOO_LARGE") from None
            if len(text) > limits.max_string:
                raise SerializationError("STRING_TOO_LONG")
            return visit({"$integer": text}, depth)
        if isinstance(obj, (float, np.floating)):
            if isinstance(obj, np.floating) and obj.dtype.itemsize > 8:
                raise SerializationError("UNSUPPORTED_FLOAT_PRECISION")
            number = float(obj)
            if not math.isfinite(number):
                raise SerializationError("NONFINITE_NUMBER")
            return number
        if isinstance(obj, Decimal):
            if not obj.is_finite():
                raise SerializationError("NONFINITE_NUMBER")
            text = str(obj)
            if len(text) > limits.max_string:
                raise SerializationError("STRING_TOO_LONG")
            return visit({"$decimal": text}, depth)
        # pandas Timestamp is a datetime subclass; handle its nanoseconds first.
        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj) or obj.tzinfo is None:
                raise SerializationError("AMBIGUOUS_TIMESTAMP")
            return obj.tz_convert("UTC").isoformat()
        if isinstance(obj, datetime):
            if obj.tzinfo is None or obj.utcoffset() is None:
                raise SerializationError("AMBIGUOUS_TIMESTAMP")
            return obj.astimezone(UTC).isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            if obj.size > limits.max_items:
                raise SerializationError("TOO_MANY_ITEMS")
            ident = id(obj)
            if ident in active:
                raise SerializationError("CYCLIC_VALUE")
            active.add(ident)
            try:
                return visit(obj.tolist(), depth)
            finally:
                active.remove(ident)
        if isinstance(obj, (dict, list, tuple)):
            if len(obj) > limits.max_items:
                raise SerializationError("TOO_MANY_ITEMS")
            ident = id(obj)
            if ident in active:
                raise SerializationError("CYCLIC_VALUE")
            active.add(ident)
            try:
                if isinstance(obj, dict):
                    if any(not isinstance(k, str) for k in obj):
                        raise SerializationError("NONSTRING_KEY")
                    return {visit(k, depth + 1): visit(v, depth + 1) for k, v in obj.items()}
                return [visit(v, depth + 1) for v in obj]
            finally:
                active.remove(ident)
        raise SerializationError("UNSUPPORTED_VALUE")

    return visit(value, 0)


def dumps(value: Any, *, limits: Limits = DEFAULT_LIMITS) -> str:
    text = json.dumps(normalize(value, limits=limits), ensure_ascii=False,
                      allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(text.encode("utf-8")) > limits.max_bytes:
        raise SerializationError("PAYLOAD_TOO_LARGE")
    return text


def loads(raw: str | bytes, *, limits: Limits = DEFAULT_LIMITS,
          legacy_nonfinite: bool = False) -> Any:
    """Strict ingress; the explicitly named legacy reader maps bare NaN to null.

    JSON string contents are never rewritten. The legacy reader is for old
    research artifacts only, whose original bytes/hash must be retained.
    """
    if not isinstance(raw, (str, bytes)):
        raise SerializationError("INVALID_JSON")
    try:
        text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
        if len(text.encode("utf-8", errors="strict")) > limits.max_bytes:
            raise SerializationError("PAYLOAD_TOO_LARGE")
    except UnicodeError:
        raise SerializationError("INVALID_UTF8") from None

    # Check nesting before the parser allocates recursive structures. JSON's
    # string escape grammar is handled by the parser; this pass only counts.
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > limits.max_depth:
                raise SerializationError("TOO_DEEP")
        elif char in "]}":
            depth -= 1

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len(items) > limits.max_items:
            raise SerializationError("TOO_MANY_ITEMS")
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SerializationError("DUPLICATE_KEY")
            result[key] = value
        return result

    def constant(_: str) -> None:
        if not legacy_nonfinite:
            raise SerializationError("NONFINITE_NUMBER")
        return None

    def floating(token: str) -> float:
        number = float(token)
        if not math.isfinite(number):
            raise SerializationError("NONFINITE_NUMBER")
        if number == 0.0:
            try:
                if Decimal(token) != 0:
                    raise SerializationError("NUMERIC_UNDERFLOW")
            except InvalidOperation:
                raise SerializationError("INVALID_NUMBER") from None
        return number

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                            parse_float=floating)
    except SerializationError:
        raise
    except (ValueError, TypeError, RecursionError):
        raise SerializationError("INVALID_JSON") from None
    # Validate bounds/types without changing Python integers on admission.
    normalize(result, limits=limits)
    return result


def public_error(code: str = "INTERNAL_ERROR") -> dict[str, str]:
    messages = {
        "INTERNAL_ERROR": "The operation could not be completed.",
        "INVALID_REQUEST": "The request does not satisfy the API contract.",
        "SERIALIZATION_ERROR": "The result could not be represented safely.",
        "BROKER_DISABLED": "Broker connectivity is disabled.",
        "DEPENDENCY_UNAVAILABLE": "An optional local dependency is unavailable.",
        "DATA_UNAVAILABLE": "No admitted data is available for this view.",
    }
    if code not in messages:
        code = "INTERNAL_ERROR"
    return {"code": code, "message": messages[code]}
