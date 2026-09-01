"""Post-processing applied after aggregation: Platt calibration, the open-vs-resolved
guard, and discrete integer snapping.

The two recalibration rails live side by side on purpose. Both take
``(prediction, question)``, both dispatch on question type internally, and both are
no-ops for the types they do not cover — so adding multiple-choice support to the guard
is a branch here rather than a new module and a new call site.
"""

import logging
from datetime import datetime

from forecasting_tools import BinaryQuestion, NumericQuestion
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.calibration import PlattParams, apply_binary_platt, apply_mc_platt
from metaculus_bot.constants import (
    PLATT_BINARY_MAX_ABS_DEVIATION,
    PLATT_CALIBRATION_ENABLED_ENV,
    PLATT_MC_MAX_ABS_DEVIATION,
    env_flag_enabled,
)
from metaculus_bot.numeric.discrete_snap import majority_votes_discrete, snap_distribution_to_integers
from metaculus_bot.resolution_state_guard import evaluate_binary, format_guard_marker

logger = logging.getLogger(__name__)


def apply_resolution_state_guard(
    prediction: PredictionTypes,
    question: MetaculusQuestion,
    now: datetime,
) -> PredictionTypes:
    """Design principle A3 — see metaculus_bot/resolution_state_guard.py for the rationale.

    Binary only today. Multiple-choice has the same failure mode through a single dominant
    option, but its clamp interacts with ballot renormalisation, so it needs its own
    threshold rather than a reused one; it lands as an ``elif`` here when it does.
    Numeric is unaffected — its analogue is interval width, which width_monitor tracks.

    Firing is normal operation, not degradation: this logs a marker and bumps no alertable
    counter, matching how the conditional-stacking counters are excluded from
    ``alertable_total``.

    ``now`` is a parameter rather than a clock read so the whole rail stays deterministic
    and testable without patching time.
    """
    # `isinstance(True, int)` is True, so bool must be excluded explicitly — it is never
    # a probability.
    if not isinstance(question, BinaryQuestion) or isinstance(prediction, bool):
        return prediction
    if not isinstance(prediction, (int, float)):
        return prediction

    verdict = evaluate_binary(question, float(prediction), now)
    if not verdict.fired:
        return prediction

    marker = format_guard_marker(verdict, question.id_of_question)
    if marker is not None:
        logger.info(marker)
    return verdict.adjusted  # type: ignore[return-value]


def apply_platt_calibration(
    prediction: PredictionTypes,
    question: MetaculusQuestion,
    binary_platt_params: PlattParams,
    mc_platt_params: PlattParams,
) -> PredictionTypes:
    """Apply post-hoc Platt scaling to a binary or MC probability.

    Returns the prediction unchanged when calibration is disabled, both param
    sets are identity, or the prediction is a NumericDistribution.
    """
    if not env_flag_enabled(PLATT_CALIBRATION_ENABLED_ENV):
        return prediction

    if binary_platt_params.is_identity() and mc_platt_params.is_identity():
        return prediction

    if isinstance(question, BinaryQuestion) and isinstance(prediction, (int, float)):
        calibrated = apply_binary_platt(
            float(prediction),
            binary_platt_params,
            max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION,
        )
        if calibrated != float(prediction):
            logger.info(
                "PLATT_BINARY: q=%s raw=%.4f calibrated=%.4f bias=%.4f slope=%.4f cap=%.3f",
                question.id_of_question,
                float(prediction),
                calibrated,
                binary_platt_params.bias,
                binary_platt_params.slope,
                PLATT_BINARY_MAX_ABS_DEVIATION,
            )
        return calibrated  # type: ignore[return-value]

    if isinstance(prediction, PredictedOptionList):
        raw_summary = {o.option_name: round(o.probability, 4) for o in prediction.predicted_options}
        apply_mc_platt(
            prediction,
            mc_platt_params,
            max_abs_deviation=PLATT_MC_MAX_ABS_DEVIATION,
        )
        calibrated_summary = {o.option_name: round(o.probability, 4) for o in prediction.predicted_options}
        if raw_summary != calibrated_summary:
            logger.info(
                "PLATT_MC: q=%s raw=%s calibrated=%s bias=%.4f slope=%.4f cap=%.3f",
                question.id_of_question,
                raw_summary,
                calibrated_summary,
                mc_platt_params.bias,
                mc_platt_params.slope,
                PLATT_MC_MAX_ABS_DEVIATION,
            )
        return prediction

    return prediction


def maybe_snap_to_integers(
    prediction: PredictionTypes,
    question: MetaculusQuestion,
    discrete_votes: list[bool],
) -> PredictionTypes:
    """Apply discrete integer CDF snapping if a majority of votes are DISCRETE.

    The caller is responsible for popping the votes from the per-question dict
    and passing them in; this function is pure.
    """
    if not isinstance(prediction, NumericDistribution) or not isinstance(question, NumericQuestion):
        return prediction

    if not majority_votes_discrete(discrete_votes):
        if discrete_votes:
            logger.info(
                "Discrete snap skipped for Q %s: votes=%s (majority=CONTINUOUS)",
                question.id_of_question,
                discrete_votes,
            )
        return prediction

    logger.info("Discrete snap: majority voted DISCRETE for Q %s | votes=%s", question.id_of_question, discrete_votes)
    snapped = snap_distribution_to_integers(prediction, question)
    if snapped is None:
        logger.info("Discrete snap returned None for Q %s (guard condition), keeping original", question.id_of_question)
        return prediction
    return snapped  # type: ignore[return-value]
