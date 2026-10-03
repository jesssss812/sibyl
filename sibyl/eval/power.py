"""Offline prospective power preflight; never calls an API."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sibyl.eval.artifacts import fingerprint, write_json
from sibyl.eval.stats import minimum_detectable_effect

ENDPOINTS = {"brier", "binary_log_score", "numeric_log_score", "mc_log_score", "crps"}


def assess_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("endpoint") not in ENDPOINTS:
        raise ValueError("Choose a defined per-type endpoint; pooled endpoint is pending")
    if not plan.get("hypothesis") or not plan.get("variance_source"):
        raise ValueError("Hypothesis and independent variance source are required")
    n, sd, effect = plan["n"], float(plan["sd"]), float(plan["effect"])
    scheduled_n = plan.get("scheduled_n")
    attrition = plan.get("attrition_allowance")
    if (
        not plan.get("manifest_id")
        or not isinstance(scheduled_n, int)
        or scheduled_n < n
        or attrition is None
        or not isinstance(attrition, (int, float))
        or not 0 <= attrition < 1
        or n > math.floor(scheduled_n * (1 - attrition))
    ):
        raise ValueError("Plan needs manifest_id, scheduled_n and feasible attrition allowance")
    if not isinstance(n, int) or n < 5 or not math.isfinite(effect) or effect <= 0 or not math.isfinite(sd) or sd <= 0:
        raise ValueError("Need n >= 5 and positive finite sd/expected effect")
    mde = minimum_detectable_effect(sd, n, power=plan.get("power", 0.8), alpha=plan.get("alpha", 0.05))
    return {
        "schema_version": 1,
        "plan": plan,
        "plan_id": fingerprint(plan),
        "assessed_at": datetime.now(timezone.utc).isoformat(),
        "mde": mde,
        "decision": "proceed" if effect >= mde else "decline",
        "assumption": "Independent question pairs; normal approximation; variance supplied prospectively",
    }


def validate_preflight(path: Path, available_n: int, manifest_id: str) -> dict[str, Any]:
    saved = json.loads(path.read_text())
    checked = assess_plan(saved["plan"])
    if checked["plan_id"] != saved.get("plan_id") or checked["decision"] != "proceed":
        raise ValueError("Missing, changed or underpowered preflight; refusing model calls")
    if saved["plan"]["scheduled_n"] != available_n or saved["plan"].get("manifest_id") != manifest_id:
        raise ValueError("Power plan does not match the frozen manifest/sample size")
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assess_plan(json.loads(args.plan.read_text()))
    write_json(args.output, result)
    print(f"{result['decision']}: prospective MDE={result['mde']:.6f}; {args.output}")
    return 0 if result["decision"] == "proceed" else 1
