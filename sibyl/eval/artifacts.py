"""Versioned, content-bound offline experiment inputs. Never serialize credentials."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion, OutOfBoundsResolution

from sibyl.eval.backtest.question_prep import BacktestQuestionSet
from sibyl.eval.backtest.scoring import GroundTruth

SCHEMA_VERSION = 1
SCORING_VERSION = "per-type-v1"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def fingerprint(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, default=str, allow_nan=False))


def code_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "sibyl"], cwd=root))
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = "unknown", True
    # Hash source bytes, never environment or git diffs that could contain secrets.
    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "sibyl").rglob("*.py"))
    }
    return {"git_sha": sha, "dirty_source": str(dirty).lower(), "source_hash": fingerprint(sources)}


def prepare_manifest(question_set: BacktestQuestionSet, *, source: str) -> dict[str, Any]:
    entries = []
    seen: set[int] = set()
    for q in question_set.questions:
        qid = q.id_of_question
        if qid is None or qid not in question_set.ground_truths:
            raise ValueError("Every manifest question requires an id and ground truth")
        if qid in seen:
            raise ValueError(f"Duplicate question id {qid}")
        seen.add(qid)
        if q.open_time is None or q.scheduled_resolution_time is None:
            raise ValueError(f"Q{qid}: historical forecast time cannot be determined")
        research = question_set.research_cache.get(qid)
        if not research or not research.strip():
            raise ValueError(f"Q{qid}: missing research; frozen replay cannot fetch live")
        as_of = q.open_time + (q.scheduled_resolution_time - q.open_time) / 2
        resolved_at = question_set.ground_truths[qid].actual_resolution_time
        if resolved_at is not None and as_of >= resolved_at:
            raise ValueError(f"Q{qid}: mid-window forecast time is after actual resolution")
        gt = asdict(question_set.ground_truths[qid])
        gt["resolution"] = question_set.ground_truths[qid].resolution
        entries.append(
            {
                "question_id": qid,
                "question_class": type(q).__name__,
                "question": q.model_dump(mode="json"),
                "ground_truth": json_safe(gt),
                "as_of": as_of.isoformat(),
                "research": research,
                "research_hash": fingerprint(research),
                "research_source": json_safe(
                    question_set.fetch_metadata.get("research_provenance", {}).get(qid, {"category": source})
                ),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "as_of_policy": "mid-window",
        "scoring_version": SCORING_VERSION,
        "preparation": json_safe(question_set.fetch_metadata),
        "entries": entries,
    }
    return {**payload, "manifest_id": fingerprint(payload), "prepared_at": datetime.now(timezone.utc).isoformat()}


def load_manifest(path: Path) -> tuple[dict[str, Any], BacktestQuestionSet]:
    data = json.loads(path.read_text())
    payload = {k: v for k, v in data.items() if k not in ("manifest_id", "prepared_at")}
    if data.get("schema_version") != SCHEMA_VERSION or fingerprint(payload) != data.get("manifest_id"):
        raise ValueError("Manifest version/hash mismatch")
    classes = {cls.__name__: cls for cls in (BinaryQuestion, NumericQuestion, DiscreteQuestion, MultipleChoiceQuestion)}
    result = BacktestQuestionSet([], {})
    for entry in data["entries"]:
        qid = entry["question_id"]
        if qid in result.ground_truths:
            raise ValueError(f"Duplicate manifest question {qid}")
        q = classes[entry["question_class"]].model_validate(entry["question"])
        if q.id_of_question != qid or fingerprint(entry["research"]) != entry["research_hash"]:
            raise ValueError(f"Q{qid}: manifest identity mismatch")
        if not entry["research"].strip():
            raise ValueError(f"Q{qid}: empty research")
        gt = dict(entry["ground_truth"])
        if gt["actual_resolution_time"]:
            gt["actual_resolution_time"] = datetime.fromisoformat(gt["actual_resolution_time"])
        if gt["resolution"] in ("above_upper_bound", "below_lower_bound"):
            gt["resolution"] = OutOfBoundsResolution(gt["resolution"])
        result.questions.append(q)
        result.ground_truths[qid] = GroundTruth(**gt)
        result.research_cache[qid] = entry["research"]
    result.fetch_metadata = {"manifest_id": data["manifest_id"], "source": str(path)}
    return data, result


def model_identity(llm: Any) -> dict[str, Any]:
    kwargs = getattr(llm, "litellm_kwargs", {})
    return {
        "model": str(getattr(llm, "model", "default")),
        "settings": {
            k: kwargs[k]
            for k in ("temperature", "max_tokens", "reasoning", "reasoning_effort", "timeout", "seed")
            if isinstance(kwargs, dict) and k in kwargs
        },
    }


def question_identity(question: Any) -> Any:
    # Production uses Pydantic snapshots. Test doubles have no serializable model_dump.
    if isinstance(question, (BinaryQuestion, NumericQuestion, MultipleChoiceQuestion)):
        value = question.model_dump(mode="json")
        if isinstance(value, dict):
            return value
    return {
        key: str(getattr(question, key, None))
        for key in ("id_of_question", "question_text", "resolution_criteria", "open_time", "scheduled_resolution_time")
    }
