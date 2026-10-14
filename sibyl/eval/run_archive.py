"""The private archive of one forecasting run, in the layout ``python -m sibyl.eval minibench-live`` scores.

``<archive-dir>/<run-id>/`` holds ``manifest.json`` (written before the first request),
``scheduled_questions.json`` (the exact questions selected, written before any is
forecast) and ``forecasts/`` (the framework's saved reports, written when the batch
returns). Nothing in it is a credential. A run ID that already has a manifest is
refused, so a second run cannot overwrite the first one's provenance.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sibyl.eval.artifacts import code_identity, question_identity

SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
SCHEDULE_FILE = "scheduled_questions.json"
FORECASTS_DIR = "forecasts"


class RunArchive:
    def __init__(self, base_dir: Path, run_id: str, configuration: dict[str, Any]) -> None:

        self.run_id = run_id
        self.run_dir = Path(base_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (self.run_dir / MANIFEST_FILE).open("x") as manifest:
                json.dump(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "code": code_identity(),
                        "configuration": configuration,
                    },
                    manifest,
                    indent=2,
                )
        except FileExistsError as exc:
            raise ValueError(f"Run ID {run_id} already has an archived manifest") from exc

    @property
    def forecasts_dir(self) -> Path:
        return self.run_dir / FORECASTS_DIR

    def record_selection(self, questions: Sequence[Any]) -> None:
        """Freeze the questions the bot is about to forecast."""

        question_ids = [question.id_of_question for question in questions]
        if any(qid is None for qid in question_ids) or len(set(question_ids)) != len(question_ids):
            raise ValueError("An archived run needs unique, non-null question IDs")
        with (self.run_dir / SCHEDULE_FILE).open("x") as snapshot:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "question_ids": question_ids,
                    "questions": [question_identity(question) for question in questions],
                },
                snapshot,
                indent=2,
            )
