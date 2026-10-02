import pytest

from sibyl.eval.power import assess_plan


def test_declines_effect_below_prospective_mde():
    result = assess_plan(
        {
            "endpoint": "brier",
            "n": 60,
            "scheduled_n": 60,
            "attrition_allowance": 0.0,
            "manifest_id": "m",
            "sd": 0.065,
            "variance_source": "independent pilot",
            "effect": 0.005,
            "hypothesis": "B reduces Brier",
            "alpha": 0.05,
            "power": 0.8,
        }
    )
    assert result["decision"] == "decline"
    assert result["mde"] > 0.02


def test_refuses_undefined_pooled_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        assess_plan({"endpoint": "pooled"})
