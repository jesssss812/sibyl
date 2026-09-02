"""What the published comment keeps, and what it drops.

The comment is the only artifact a human reader ever sees, and it is also the archive
`eval/performance` parses months later. So the stripper has two jobs that pull against
each other: remove what is operator telemetry, and touch nothing that is a parsed
contract. The second half is the one worth testing hardest -- deleting a section that
reads as noise but is silently load-bearing is exactly the mistake available here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sibyl.publishing.comment.formatting import strip_internal_sections

_MINI_FIXTURE = Path(__file__).parents[1] / "data" / "performance_comments_mini.jsonl"

_COMMENT = """
# SUMMARY
*Question*: Will X happen?
*Final Prediction*: 54.5%
*Total Cost*: extra_metadata_in_explanation is disabled
*Time Spent*: extra_metadata_in_explanation is disabled
*LLMs*: `extra_metadata_in_explanation is disabled`
*Bot Name*: extra_metadata_in_explanation is disabled


## Report 1 Summary
### Forecasts
*Forecaster 1 (alpha)*: 68.22%
*Forecaster 2 (beta)*: 54.46%


# RESEARCH
## Report 1 Research
### Research (AskNews)
Some research prose that explains the reasoning.

---

### Provider Diagnostics

- asknews: ok | 4210 chars | 1200 ms
- native_search: lost=grounding:0 | 0 chars | 300 ms

# FORECASTS
## R1: Forecaster 1 Reasoning
Model: openrouter/openai/gpt-5.6-sol

Prose reasoning a reader wants.

── STRUCTURED FORECAST (machine-readable; REQUIRED) ──

```json
{"question_type": "binary", "posterior_prob": 0.6822}
```
<!-- STACKER_OUTCOME=primary -->
<!-- STACKED=true -->
<!-- FORECASTERS_USED=2/3 -->
"""


class TestWhatIsRemoved:
    @pytest.mark.parametrize("field", ["Total Cost", "Time Spent", "LLMs", "Bot Name"])
    def test_the_placeholder_metadata_lines_go(self, field: str) -> None:
        """Four lines of literally zero information, above the forecast, in every comment."""
        assert f"*{field}*" not in strip_internal_sections(_COMMENT)

    def test_the_placeholder_text_appears_nowhere(self) -> None:
        assert "extra_metadata_in_explanation" not in strip_internal_sections(_COMMENT)

    def test_the_provider_diagnostics_block_goes(self) -> None:
        out = strip_internal_sections(_COMMENT)
        assert "Provider Diagnostics" not in out
        assert "asknews: ok | 4210 chars" not in out
        assert "lost=grounding" not in out

    def test_real_metadata_survives_if_it_is_ever_enabled(self) -> None:
        """Matched on the placeholder, not the field name.

        Turning extra_metadata_in_explanation on should show the real values, not have
        them silently deleted by a stripper written while it was off.
        """
        enabled = _COMMENT.replace(
            "*Total Cost*: extra_metadata_in_explanation is disabled", "*Total Cost*: $0.42 (estimated)"
        )
        assert "*Total Cost*: $0.42 (estimated)" in strip_internal_sections(enabled)


class TestWhatMustSurvive:
    def test_the_structured_forecast_block_survives(self) -> None:
        """The load-bearing one.

        Since 2026-07 the fenced JSON block is the ONLY place per-model forecast values
        appear in a comment. `eval/performance/parsing.py` recovers them from the archive
        through it, so dropping it would silently end per-model attribution for every
        comment published afterwards.
        """
        out = strip_internal_sections(_COMMENT)
        assert "STRUCTURED FORECAST" in out
        assert '"posterior_prob": 0.6822' in out

    @pytest.mark.parametrize(
        "marker", ["<!-- STACKER_OUTCOME=primary -->", "<!-- STACKED=true -->", "<!-- FORECASTERS_USED=2/3 -->"]
    )
    def test_the_html_markers_survive(self, marker: str) -> None:
        # A parsed contract, and invisible in rendered markdown, so there is no reader-facing
        # reason to remove them either.
        assert marker in strip_internal_sections(_COMMENT)

    def test_the_per_forecaster_summary_survives(self) -> None:
        out = strip_internal_sections(_COMMENT)
        assert "*Forecaster 1 (alpha)*: 68.22%" in out
        assert "*Forecaster 2 (beta)*: 54.46%" in out

    def test_the_research_prose_survives(self) -> None:
        assert "Some research prose that explains the reasoning." in strip_internal_sections(_COMMENT)

    def test_the_reasoning_prose_survives(self) -> None:
        assert "Prose reasoning a reader wants." in strip_internal_sections(_COMMENT)

    def test_the_headline_prediction_survives(self) -> None:
        out = strip_internal_sections(_COMMENT)
        assert "*Final Prediction*: 54.5%" in out and "*Question*: Will X happen?" in out


class TestShape:
    def test_sections_keep_a_blank_line_before_the_next_heading(self) -> None:
        # The diagnostics block runs up to the next top-level heading; swallowing its
        # trailing blank line would butt the research text against "# FORECASTS".
        assert "\n\n# FORECASTS" in strip_internal_sections(_COMMENT)

    def test_no_run_of_three_or_more_blank_lines(self) -> None:
        assert "\n\n\n" not in strip_internal_sections(_COMMENT)

    def test_it_actually_gets_shorter(self) -> None:
        assert len(strip_internal_sections(_COMMENT)) < len(_COMMENT)

    def test_is_idempotent(self) -> None:
        once = strip_internal_sections(_COMMENT)
        assert strip_internal_sections(once) == once


class TestAgainstRealArchivedComments:
    """The archive is what the parsers actually run on, so it is what must not break."""

    @pytest.fixture(scope="class")
    def comments(self) -> list[str]:
        records = [json.loads(line) for line in _MINI_FIXTURE.read_text().splitlines() if line.strip()]
        return [record["comment_text"] for record in records]

    def test_stripping_preserves_every_parsed_value(self, comments: list[str]) -> None:
        """Run all six parsers before and after. Not one output may change.

        These twelve records were selected to cover one distinct parser-relevant comment
        shape each, so agreement across all of them is a real check rather than a lucky one.
        """
        from sibyl.eval.performance.parsing import (
            parse_forecaster_model_map,
            parse_inferred_stacker_outcome,
            parse_per_model_forecasts,
            parse_per_model_mc_option_probs,
            parse_per_model_numeric_percentiles,
            parse_stacked_marker,
        )

        parsers = [
            parse_per_model_forecasts,
            parse_forecaster_model_map,
            parse_per_model_numeric_percentiles,
            parse_per_model_mc_option_probs,
            parse_inferred_stacker_outcome,
            parse_stacked_marker,
        ]
        assert comments, "fixture is empty"
        for comment in comments:
            stripped = strip_internal_sections(comment)
            for parser in parsers:
                assert parser(stripped) == parser(comment), f"{parser.__name__} changed after stripping"
