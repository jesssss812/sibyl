"""Disk-cache foundation for the probabilistic-tools ablation benchmark.

The cache lives at ``backtests/ablation/`` (overridable for tests) and is
organized per pipeline stage:

* ``qids.json`` — question manifest mapping ``qid`` → metadata dict.
* ``research/<qid>.md`` + ``research/<qid>.meta.json`` — concatenated research
  blob (markdown sidecar) and provider metadata (JSON).
* ``leakage_screens/<qid>.json`` — leakage detector verdict.
* ``forecaster_outputs/<qid>/<model_slug>.json`` — per-model rationale + parsed
  prediction value.
* ``stacker_outputs/<qid>/arm_<arm>.json`` — per-arm output. Deterministic arms
  (median / mean / pdf_*) use this stacker-independent filename. The LLM-stacker
  arms (stack / stack_aug) pass a ``stacker_slug`` and land at
  ``arm_<arm>__<stacker_slug>.json`` so a stacker swap never clobbers another
  stacker's results while deterministic arms stay shared.
* ``scores/run_<ts>.json`` + ``scores/summary_<ts>.md`` — per-run analysis.

Each individual JSON write is atomic: the payload is written to a tempfile
in the same parent directory and ``os.replace``'d into place, so a crash
mid-write to a single file leaves it in either the old state or the new
state — never partially corrupt.

Multi-file pairs (meta+blob in ``write_research`` / ``write_pruned_research``)
are NOT atomic across both files. ``write_research`` writes meta FIRST, blob
SECOND; an interruption between the two leaves a partial state with the
NEWER meta on disk and the OLDER blob (rather than the silently-corrupt
inverse, NEW blob + OLD meta). The meta includes a blob hash. Readers verify the pair and treat an interrupted
write as a cache miss, so the producing stage must run again.

JSON payloads carry a top-level ``cache_schema_version`` field. Reads raise
``ValueError`` on a mismatch instead of returning silently-stale data; writes
auto-inject the current version so callers don't need to remember.
"""

import json
import os
import tempfile
from pathlib import Path

from sibyl.eval.artifacts import code_identity, fingerprint, json_safe

CACHE_SCHEMA_VERSION = 1

_DEFAULT_ROOT = "backtests/ablation"


def model_slug_to_filename(model: str) -> str:
    """Turn an OpenRouter model identifier into a filesystem-safe slug.

    Strips an ``openrouter/`` prefix, then replaces ``/`` and ``:`` with ``__``.
    Dots are kept (they appear in version strings like ``minimax-m2.5``).
    Leading/trailing whitespace is stripped.
    """
    cleaned = model.strip()
    if cleaned.startswith("openrouter/"):
        cleaned = cleaned[len("openrouter/") :]
    return cleaned.replace("/", "__").replace(":", "__")


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via a same-directory tempfile.

    Public surface for callers outside ``cache.py``. ``manual_rejects.json``,
    ``qa_summary_*.md``, ``qa_reports/<qid>.json``, and ``qa_research_*.md``
    all flow through this helper so a crash mid-write never corrupts the
    operator-visible recovery surface.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# Backwards-compatible alias for the previous private name.
_atomic_write_text = atomic_write_text


def _dump_json(payload: dict) -> str:
    return json.dumps(json_safe(payload), indent=2, sort_keys=True, default=str, allow_nan=False)


def _load_versioned_json(path: Path) -> dict:
    """Load a JSON payload and verify ``cache_schema_version`` matches."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    on_disk_version = payload.get("cache_schema_version")
    if on_disk_version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"cache_schema_version mismatch in {path}: expected {CACHE_SCHEMA_VERSION}, got {on_disk_version}"
        )
    return payload


def _inject_version(payload: dict) -> dict:
    return {**payload, "cache_schema_version": CACHE_SCHEMA_VERSION}


class AblationCache:
    """Filesystem-backed cache for the probabilistic-tools ablation benchmark."""

    def __init__(self, root: Path | str = _DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self._input_hashes: dict[tuple[str, int, str], str] = {}
        self._code_hash: str | None = None
        self.valid_hits: dict[str, int] = {}
        self.verified_reads: list[tuple[str, int, str]] = []

    def bind_inputs(self, stage: str, qid: int, key: str, inputs: dict) -> None:
        """Bind reads and writes to the same computation, including its implementation."""
        if self._code_hash is None:
            self._code_hash = code_identity()["source_hash"]
        self._input_hashes[(stage, qid, key)] = fingerprint({"code": self._code_hash, "inputs": inputs})

    def _verified(self, payload: dict, stage: str, qid: int, key: str) -> dict | None:
        expected = self._input_hashes.get((stage, qid, key))
        if expected is not None and payload.get("input_hash") != expected:
            return None
        if expected is not None and not (stage == "forecaster" and
                                        (payload.get("prediction_value") is None or payload.get("errors") or payload.get("error"))):
            self.valid_hits[stage] = self.valid_hits.get(stage, 0) + 1
            self.verified_reads.append((stage, qid, key))
        return payload

    def _stamped(self, payload: dict, stage: str, qid: int, key: str) -> dict:
        expected = self._input_hashes.get((stage, qid, key))
        return {**payload, "input_hash": expected} if expected is not None else payload

    # ------------------------------------------------------------------
    # qids manifest
    # ------------------------------------------------------------------

    @property
    def _qids_path(self) -> Path:
        return self.root / "qids.json"

    def read_qids_manifest(self) -> dict[int, dict]:
        if not self._qids_path.exists():
            return {}
        payload = _load_versioned_json(self._qids_path)
        entries = payload["entries"]
        return {int(qid): metadata for qid, metadata in entries.items()}

    def write_qids_manifest(self, manifest: dict[int, dict]) -> None:
        entries = {str(qid): metadata for qid, metadata in manifest.items()}
        payload = _inject_version({"entries": entries})
        _atomic_write_text(self._qids_path, _dump_json(payload))

    def append_qids_manifest(self, new_entries: dict[int, dict]) -> dict[int, dict]:
        existing = self.read_qids_manifest()
        merged = {**existing, **new_entries}
        self.write_qids_manifest(merged)
        return merged

    # ------------------------------------------------------------------
    # Research (markdown blob + meta JSON)
    # ------------------------------------------------------------------

    def _research_blob_path(self, qid: int) -> Path:
        return self.root / "research" / f"{qid}.md"

    def _research_meta_path(self, qid: int) -> Path:
        return self.root / "research" / f"{qid}.meta.json"

    def read_research(self, qid: int) -> tuple[str, dict] | None:
        blob_path = self._research_blob_path(qid)
        meta_path = self._research_meta_path(qid)
        if not blob_path.exists() or not meta_path.exists():
            return None
        blob = blob_path.read_text(encoding="utf-8")
        meta = _load_versioned_json(meta_path)
        if meta.get("blob_hash") != fingerprint(blob):
            return None
        return blob, meta

    def write_research(self, qid: int, blob: str, meta: dict) -> None:
        # Write meta FIRST, blob SECOND. Each individual atomic_write_text is
        # atomic, but the pair is not — interrupting between leaves a partial
        # state. Writing meta first means a partial state has the newer meta
        # on disk with the OLD blob (caller can detect via stale data signals
        # or re-run the producing stage), rather than the silently-corrupt
        # state of NEW blob + OLD meta.
        _atomic_write_text(self._research_meta_path(qid), _dump_json(_inject_version({**meta, "blob_hash": fingerprint(blob)})))
        _atomic_write_text(self._research_blob_path(qid), blob)

    def has_research(self, qid: int) -> bool:
        return self.read_research(qid) is not None

    # ------------------------------------------------------------------
    # Pruned research (sanitized blob written by the redactor subagent)
    # ------------------------------------------------------------------

    def _pruned_research_blob_path(self, qid: int) -> Path:
        return self.root / "research_pruned" / f"{qid}.md"

    def _pruned_research_meta_path(self, qid: int) -> Path:
        return self.root / "research_pruned" / f"{qid}.meta.json"

    def read_pruned_research(self, qid: int) -> tuple[str, dict] | None:
        blob_path = self._pruned_research_blob_path(qid)
        meta_path = self._pruned_research_meta_path(qid)
        if not blob_path.exists() or not meta_path.exists():
            return None
        blob = blob_path.read_text(encoding="utf-8")
        meta = _load_versioned_json(meta_path)
        if meta.get("blob_hash") != fingerprint(blob):
            return None
        return blob, meta

    def write_pruned_research(self, qid: int, sanitized_blob: str, meta: dict) -> None:
        # Same M2 invariant as write_research: meta first, blob second.
        _atomic_write_text(self._pruned_research_meta_path(qid), _dump_json(_inject_version({**meta, "blob_hash": fingerprint(sanitized_blob)})))
        _atomic_write_text(self._pruned_research_blob_path(qid), sanitized_blob)

    def has_pruned_research(self, qid: int) -> bool:
        return self.read_pruned_research(qid) is not None

    # ------------------------------------------------------------------
    # Leakage screens
    # ------------------------------------------------------------------

    def _leakage_screen_path(self, qid: int) -> Path:
        return self.root / "leakage_screens" / f"{qid}.json"

    def read_leakage_screen(self, qid: int) -> dict | None:
        path = self._leakage_screen_path(qid)
        if not path.exists():
            return None
        return _load_versioned_json(path)

    def write_leakage_screen(self, qid: int, payload: dict) -> None:
        _atomic_write_text(self._leakage_screen_path(qid), _dump_json(_inject_version(payload)))

    # ------------------------------------------------------------------
    # Forecaster outputs (per-model)
    # ------------------------------------------------------------------

    def _forecaster_output_path(self, qid: int, model_slug: str) -> Path:
        return self.root / "forecaster_outputs" / str(qid) / f"{model_slug}.json"

    def read_forecaster_output(self, qid: int, model_slug: str) -> dict | None:
        path = self._forecaster_output_path(qid, model_slug)
        if not path.exists():
            return None
        return self._verified(_load_versioned_json(path), "forecaster", qid, model_slug)

    def write_forecaster_output(self, qid: int, model_slug: str, payload: dict) -> None:
        _atomic_write_text(
            self._forecaster_output_path(qid, model_slug),
            _dump_json(_inject_version(self._stamped(payload, "forecaster", qid, model_slug))),
        )

    def list_forecaster_outputs(
        self,
        qid: int,
        lineup_filter: list[str] | None = None,
    ) -> dict[str, dict]:
        """Return ``{filename_stem: payload}`` for every forecaster output on disk.

        When ``lineup_filter`` is provided, only entries whose stem matches
        ``model_slug_to_filename(m)`` for some ``m`` in the filter are kept.
        Use this to ignore obsolete-lineup leftovers when checking forecast
        cache freshness.
        """
        directory = self.root / "forecaster_outputs" / str(qid)
        if not directory.exists():
            return {}
        outputs: dict[str, dict] = {}
        for path in sorted(directory.glob("*.json")):
            outputs[path.stem] = _load_versioned_json(path)
        if lineup_filter is not None:
            allowed = {model_slug_to_filename(m) for m in lineup_filter}
            outputs = {stem: payload for stem, payload in outputs.items() if stem in allowed}
        return outputs

    # ------------------------------------------------------------------
    # Stacker outputs (per arm)
    # ------------------------------------------------------------------

    def _stacker_output_path(self, qid: int, arm: str, stacker_slug: str | None = None) -> Path:
        """Path for one arm's cached output.

        ``stacker_slug=None`` keeps the original ``arm_<arm>.json`` filename —
        this is the back-compat path used by deterministic arms (median / mean /
        pdf_*), forecaster-shared outputs, and every pre-existing on-disk file.
        When ``stacker_slug`` is supplied (the LLM-stacker arms), the filename
        becomes ``arm_<arm>__<stacker_slug>.json`` so two runs with different
        stackers never overwrite each other's results.
        """
        directory = self.root / "stacker_outputs" / str(qid)
        if stacker_slug is None:
            return directory / f"arm_{arm}.json"
        return directory / f"arm_{arm}__{stacker_slug}.json"

    def read_stacker_output(self, qid: int, arm: str, stacker_slug: str | None = None) -> dict | None:
        path = self._stacker_output_path(qid, arm, stacker_slug)
        if not path.exists():
            return None
        return self._verified(_load_versioned_json(path), "stacker", qid, arm + (stacker_slug or ""))

    def write_stacker_output(self, qid: int, arm: str, payload: dict, stacker_slug: str | None = None) -> None:
        _atomic_write_text(
            self._stacker_output_path(qid, arm, stacker_slug),
            _dump_json(_inject_version(self._stamped(payload, "stacker", qid, arm + (stacker_slug or "")))),
        )

    # ------------------------------------------------------------------
    # Score runs
    # ------------------------------------------------------------------

    def write_score_run(self, run_id: str, payload: dict) -> Path:
        path = self.root / "scores" / f"run_{run_id}.json"
        _atomic_write_text(path, _dump_json(_inject_version(payload)))
        return path

    def write_score_summary(self, run_id: str, markdown: str) -> Path:
        path = self.root / "scores" / f"summary_{run_id}.md"
        _atomic_write_text(path, markdown)
        return path
