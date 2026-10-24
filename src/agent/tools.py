"""Tools the agent may call: a JSON schema for the arguments, a timeout, and a flag for
side effects.

A tool's arguments come from a model, so they are validated here (the model's schema
adherence is not trusted), and its results go back to the model as **data**: document text
is escaped so it cannot close or forge the result element. A tool with ``side_effect=True``
is never run by the agent loop directly; the run pauses until a person approves the call.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from generation.prompt import citation_label

MAX_RESULT_CHARS = 4000  # a tool result longer than this is cut, so one call cannot fill the context


class ToolError(Exception):
    """An expected failure the model should see and may recover from (bad lot number,
    unknown alloy). Unexpected exceptions are reported without their message."""


@dataclass
class ToolContext:
    run_id: str
    call_id: str
    add_sources: Callable[[List[dict]], List[int]] = lambda sources: []


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[[Dict[str, Any], ToolContext], str]
    side_effect: bool = False
    timeout_s: float = 10.0

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def validate(schema: Dict[str, Any], value: Any, path: str = "arguments") -> List[str]:
    """Check ``value`` against the subset of JSON Schema the tools use: object
    (properties, required, additionalProperties), string (minLength, maxLength,
    pattern, enum), integer and number (minimum, maximum), boolean. Returns the errors."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path} must be an object"]
        errors = [f"{path}.{name} is required" for name in schema.get("required", []) if name not in value]
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                errors += validate(properties[name], item, f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{name} is not allowed")
        return errors
    if kind == "string":
        if not isinstance(value, str):
            return [f"{path} must be a string"]
        errors = []
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path} is longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path} does not match {schema['pattern']}")
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path} must be one of {schema['enum']}")
        return errors
    if kind in ("integer", "number"):
        # bool is an int in Python, but true is not a valid count
        numeric = isinstance(value, int) if kind == "integer" else isinstance(value, (int, float))
        if isinstance(value, bool) or not numeric:
            return [f"{path} must be an {kind}"]
        errors = []
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be at most {schema['maximum']}")
        return errors
    if kind == "boolean":
        return [] if isinstance(value, bool) else [f"{path} must be a boolean"]
    return []


def escape(text: str) -> str:
    """Tool output is data: it must not be able to close its element or open a fake one."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


# ---- read-only tools -----------------------------------------------------------------


def search_tool(index, max_top_k: int = 5) -> Tool:
    def run(args: Dict[str, Any], ctx: ToolContext) -> str:
        results = index.search(args["query"], top_k=args.get("top_k", 3))
        if not results:
            return "No passages found."
        sources = [{"label": citation_label(r), "source": str(r.source), "text": r.text} for r in results]
        numbers = ctx.add_sources(sources)
        return "\n".join(
            f'<result id="{n}" label="{escape(s["label"])}">{escape(s["text"])}</result>' for n, s in zip(numbers, sources)
        )

    return Tool(
        name="search_documents",
        description=(
            "Search the engineering documents (diaphragm forming, Co-Cr-Ni-Mo alloys, heat treatment, inspection, "
            "quality records). Returns passages with an id; cite a passage as [id] in the answer."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 300, "description": "What to look for, in plain words"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": max_top_k, "description": "Number of passages (default 3)"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        func=run,
    )


def alloy_tool(table: Union[str, Path]) -> Tool:
    """Exact lookup in the alloy table (a CSV): structured facts come from the table,
    not from searching text."""
    with Path(table).open(encoding="utf-8", newline="") as handle:
        rows = {row["Alloy"].strip().upper(): row for row in csv.DictReader(handle)}

    def run(args: Dict[str, Any], ctx: ToolContext) -> str:
        row = rows.get(args["alloy"].upper())
        if row is None:  # unreachable after validation, kept for direct calls
            raise ToolError(f"unknown alloy {args['alloy']!r}")
        return escape(json.dumps(row, ensure_ascii=False))

    return Tool(
        name="lookup_alloy",
        description="Look up one alloy in the alloy table: base element, product form, notable property, typical use.",
        parameters={
            "type": "object",
            "properties": {"alloy": {"type": "string", "enum": sorted(rows), "description": "Alloy name"}},
            "required": ["alloy"],
            "additionalProperties": False,
        },
        func=run,
    )


# ---- a tool with a side effect -------------------------------------------------------


class NcrStore:
    """Nonconformance reports in an append-only JSONL file: a stand-in for a quality
    system. ``create`` is idempotent per key, so approving the same call twice (a retry,
    a double click) never files two reports."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._lock = threading.Lock()  # check-then-append must not interleave

    def _all(self) -> List[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def find(self, key: str) -> Optional[dict]:
        return next((r for r in self._all() if r.get("idempotency_key") == key), None)

    def create(self, key: str, fields: Dict[str, Any]) -> dict:
        with self._lock:
            return self._create(key, fields)

    def _create(self, key: str, fields: Dict[str, Any]) -> dict:
        existing = self.find(key)
        if existing:
            return existing
        record = {
            "ncr_id": "NCR-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8].upper(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "idempotency_key": key,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def list(self) -> List[dict]:
        return self._all()


def ncr_tool(store: NcrStore) -> Tool:
    def run(args: Dict[str, Any], ctx: ToolContext) -> str:
        record = store.create(f"{ctx.run_id}:{ctx.call_id}", {k: args[k] for k in ("lot", "defect", "root_cause", "corrective_action")})
        return f"Nonconformance report {record['ncr_id']} raised for lot {escape(record['lot'])}."

    text = {"type": "string", "minLength": 3, "maxLength": 500}
    return Tool(
        name="raise_nonconformance_report",
        description=(
            "Raise a nonconformance report (NCR) for a production lot that failed inspection. This creates a record "
            "in the quality system, so a person must approve it first. Only use it when the user asks for a report."
        ),
        parameters={
            "type": "object",
            "properties": {
                "lot": {"type": "string", "pattern": r"^[A-Z]{1,3}-?\d{1,8}$", "description": "Lot number, e.g. L-302"},
                "defect": {**text, "description": "What failed, as observed"},
                "root_cause": {**text, "description": "Likely root cause, based on the documents"},
                "corrective_action": {**text, "description": "Proposed corrective action"},
            },
            "required": ["lot", "defect", "root_cause", "corrective_action"],
            "additionalProperties": False,
        },
        func=run,
        side_effect=True,
    )


def default_tools(index, alloy_table: Union[str, Path, None], ncr_store: NcrStore) -> List[Tool]:
    tools = [search_tool(index)]
    if alloy_table and Path(alloy_table).exists():
        tools.append(alloy_tool(alloy_table))
    tools.append(ncr_tool(ncr_store))
    return tools


__all__ = ["MAX_RESULT_CHARS", "NcrStore", "Tool", "ToolContext", "ToolError", "alloy_tool", "default_tools", "escape",
           "ncr_tool", "search_tool", "validate"]
