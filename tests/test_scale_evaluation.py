import pytest

from leibniz.scale_evaluation import (
    AdaptiveScaleEvaluation,
    PerScaleScore,
    ScaleAxis,
    ScaleEvaluationLevel,
    ScaleEvaluationTrace,
)


def test_scale_trace_integrates_consecutive_local_competence() -> None:
    trace = ScaleEvaluationTrace(
        axis=ScaleAxis(symbol="L", minimum=1),
        score=PerScaleScore(),
        evaluation=AdaptiveScaleEvaluation(axis_symbol="L"),
        levels=(
            ScaleEvaluationLevel(scale=1, competence=0.75, score_weight=1.0),
            ScaleEvaluationLevel(
                scale=2,
                competence=0.5,
                score_weight=2.0,
            ),
            ScaleEvaluationLevel(
                scale=3,
                competence=0.0,
                score_weight=3.0,
                boundary_reason="model output_shape does not match resolved outcome space",
            ),
        ),
        stop_reason="model-scale-boundary",
    )

    assert trace.integrated_score == 1.75
    assert ScaleEvaluationTrace.from_record(trace.to_record()) == trace


def test_adaptive_scale_evaluation_stops_on_zero_marginal_score() -> None:
    evaluation = AdaptiveScaleEvaluation(axis_symbol="L", stopping_window=2)

    assert not evaluation.should_stop((1.0, 0.0))
    assert evaluation.should_stop((1.0, 0.0, 0.0))


def test_scale_trace_requires_consecutive_scales_from_minimum() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        ScaleEvaluationTrace(
            axis=ScaleAxis(symbol="L", minimum=1),
            score=PerScaleScore(),
            evaluation=AdaptiveScaleEvaluation(axis_symbol="L"),
            levels=(
                ScaleEvaluationLevel(scale=1, competence=0.75, score_weight=1.0),
                ScaleEvaluationLevel(scale=3, competence=0.25, score_weight=3.0),
            ),
            stop_reason="test",
        )
