"""Score-stage refusal uses the complete scheduled set for each comparison."""
from sibyl.eval.ablation.cache import AblationCache
from sibyl.eval.ablation.cli import _build_parser, _stage_score
from sibyl.eval.ablation.run_stacker import ARM_MEDIAN
from tests.unit.test_ablation_cli import _build_six_arm_working_set


def test_more_than_five_percent_missing_refuses(tmp_path):
    working = _build_six_arm_working_set(list(range(1, 21)))
    del working.stacker_payloads[ARM_MEDIAN][1]
    del working.stacker_payloads[ARM_MEDIAN][2]
    args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(tmp_path)])
    path = _stage_score(args, AblationCache(tmp_path), working)
    assert working.inference_refused
    assert path.read_text().startswith("# REFUSED")


def test_exactly_five_percent_does_not_trigger_drop_refusal(tmp_path):
    working = _build_six_arm_working_set(list(range(1, 21)))
    del working.stacker_payloads[ARM_MEDIAN][1]
    args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(tmp_path)])
    _stage_score(args, AblationCache(tmp_path), working)
    assert not working.inference_refused
