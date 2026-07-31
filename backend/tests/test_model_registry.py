"""Tests for the churn model registry.

The point of these tests is that **a broken model file must not take the API
down**. Each way an artifact can be wrong gets its own case, and every one of
them must surface as a typed :class:`ModelNotLoadedError` (which the global
handler renders as a 503) rather than an unhandled exception.

The fixtures build their own miniature model instead of leaning on the real
trained artifact: ``backend/app/ml/artifacts/`` is gitignored, so on a fresh
clone or in CI it does not exist. A test suite that silently passes only on the
author's machine is worse than no test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelNotLoadedError
from app.ml import registry
from app.ml.features import FEATURE_NAMES


@pytest.fixture(autouse=True)
def _clear_cache():
    """Registry state is module-level, so isolate every test from its neighbours."""
    registry.reset()
    yield
    registry.reset()


@pytest.fixture
def artifact_dir(tmp_path, monkeypatch) -> Path:
    """Point the registry at an empty throwaway artifact directory.

    ``get_settings`` is patched in the registry's own namespace rather than via
    environment variables: the settings object is ``lru_cache``d, so an env var
    set mid-process would not reliably take effect.
    """
    directory = tmp_path / "artifacts"
    directory.mkdir()
    fake = SimpleNamespace(
        model_file=directory / "churn_model.joblib",
        model_metadata_file=directory / "model_metadata.json",
    )
    monkeypatch.setattr(registry, "get_settings", lambda: fake)
    return directory


def _tiny_pipeline() -> Pipeline:
    """A real (if unimpressive) fitted classifier over the true feature contract."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, len(FEATURE_NAMES)))
    # Make the label depend on recency so the model learns a usable direction.
    labels = (features[:, 0] > 0).astype(int)
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression())]
    )
    pipeline.fit(pd.DataFrame(features, columns=FEATURE_NAMES), labels)
    return pipeline


def _write_artifact(
    directory: Path,
    *,
    pipeline=None,
    bundle_features: list[str] | None = None,
    card_features: list[str] | None = None,
    bare: bool = False,
) -> None:
    """Persist a model bundle + model card in the shape ``train.py`` produces."""
    pipeline = _tiny_pipeline() if pipeline is None else pipeline
    payload = (
        pipeline
        if bare
        else {
            "pipeline": pipeline,
            "feature_names": list(
                FEATURE_NAMES if bundle_features is None else bundle_features
            ),
            "model_version": "1.0.0",
            "trained_at": "2026-01-01T00:00:00+00:00",
        }
    )
    joblib.dump(payload, directory / "churn_model.joblib")
    (directory / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_version": "1.0.0",
                "features": list(
                    FEATURE_NAMES if card_features is None else card_features
                ),
                "metrics": {"roc_auc": 0.8},
            }
        ),
        encoding="utf-8",
    )


def _sample_features() -> dict[str, float]:
    return {name: 1.0 for name in FEATURE_NAMES}


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_loads_bundle_and_exposes_feature_order(artifact_dir):
    _write_artifact(artifact_dir)

    model = registry.load_model()

    assert model.features == FEATURE_NAMES
    assert model.version == "1.0.0"
    assert hasattr(model.pipeline, "predict_proba")


def test_accepts_a_bare_estimator_for_forward_compatibility(artifact_dir):
    """A future training script might persist the estimator without a wrapper."""
    _write_artifact(artifact_dir, bare=True)

    model = registry.load_model()

    # Falls back to the model card's feature list when the bundle has none.
    assert model.features == FEATURE_NAMES


def test_predict_proba_returns_probabilities(artifact_dir):
    _write_artifact(artifact_dir)

    probabilities = registry.predict_proba(_sample_features())

    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0


def test_predict_proba_scores_a_batch(artifact_dir):
    _write_artifact(artifact_dir)
    frame = pd.DataFrame([_sample_features(), _sample_features()])

    assert registry.predict_proba(frame).shape == (2,)


def test_predict_proba_is_insensitive_to_column_order(artifact_dir):
    """The pipeline is positional, so the registry must reorder, not trust callers."""
    _write_artifact(artifact_dir)
    features = _sample_features()
    features["recency"] = 400.0  # a value that will move the score

    shuffled = dict(reversed(list(features.items())))
    expected = registry.predict_proba(pd.DataFrame([features], columns=FEATURE_NAMES))

    assert registry.predict_proba(shuffled)[0] == pytest.approx(expected[0])


def test_metadata_is_exposed_for_the_model_performance_endpoint(artifact_dir):
    _write_artifact(artifact_dir)

    assert registry.get_metadata()["metrics"]["roc_auc"] == 0.8


# --------------------------------------------------------------------------
# Failure paths - each must degrade to a typed 503, never crash
# --------------------------------------------------------------------------


def test_missing_artifact_raises_typed_error_with_remediation(artifact_dir):
    with pytest.raises(ModelNotLoadedError) as exc_info:
        registry.load_model()

    error = exc_info.value
    assert error.status == 503
    assert error.code == "MODEL_NOT_LOADED"
    # The message should tell a reviewer how to fix it, not just what broke.
    assert "train_churn_model.py" in error.message


def test_missing_metadata_alone_is_still_an_error(artifact_dir):
    _write_artifact(artifact_dir)
    (artifact_dir / "model_metadata.json").unlink()

    with pytest.raises(ModelNotLoadedError):
        registry.load_model()


def test_zero_length_artifact_is_treated_as_corrupt(artifact_dir):
    """An interrupted training run leaves a 0-byte file behind."""
    (artifact_dir / "churn_model.joblib").touch()
    (artifact_dir / "model_metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelNotLoadedError, match="empty"):
        registry.load_model()


def test_unpicklable_artifact_is_reported_as_corrupt(artifact_dir):
    (artifact_dir / "churn_model.joblib").write_bytes(b"this is not a pickle")
    (artifact_dir / "model_metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelNotLoadedError) as exc_info:
        registry.load_model()

    assert exc_info.value.status == 503


def test_bundle_without_pipeline_key_is_rejected(artifact_dir):
    joblib.dump({"scaler_only": 1}, artifact_dir / "churn_model.joblib")
    (artifact_dir / "model_metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelNotLoadedError, match="pipeline"):
        registry.load_model()


def test_non_classifier_payload_is_rejected(artifact_dir):
    """Something that unpickles fine but cannot produce probabilities."""
    _write_artifact(artifact_dir, pipeline=StandardScaler())

    with pytest.raises(ModelNotLoadedError, match="predict_proba"):
        registry.load_model()


def test_invalid_metadata_json_is_rejected(artifact_dir):
    _write_artifact(artifact_dir)
    (artifact_dir / "model_metadata.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ModelNotLoadedError):
        registry.load_model()


def test_feature_order_mismatch_between_bundle_and_card_is_rejected(artifact_dir):
    """Two files from different training runs must not be silently combined."""
    _write_artifact(
        artifact_dir, card_features=list(reversed(FEATURE_NAMES))
    )

    with pytest.raises(ModelNotLoadedError, match="disagree"):
        registry.load_model()


def test_stale_feature_set_is_rejected_rather_than_mis_scoring(artifact_dir):
    """The worst failure mode: confident predictions from misaligned coefficients."""
    stale = FEATURE_NAMES[:-1]
    _write_artifact(artifact_dir, bundle_features=stale, card_features=stale)

    with pytest.raises(ModelNotLoadedError) as exc_info:
        registry.load_model()

    assert exc_info.value.details["expected_features"] == FEATURE_NAMES


def test_scoring_without_required_features_is_rejected(artifact_dir):
    """Guards the case where a caller bypasses the request schema."""
    _write_artifact(artifact_dir)

    with pytest.raises(ModelNotLoadedError, match="missing features"):
        registry.predict_proba({"recency": 10.0})


# --------------------------------------------------------------------------
# Availability probing and caching
# --------------------------------------------------------------------------


def test_is_available_reports_false_without_raising(artifact_dir):
    """``/api/health`` must be able to report degradation without erroring."""
    assert registry.is_available() is False

    _write_artifact(artifact_dir)
    registry.reset()

    assert registry.is_available() is True


def test_model_is_deserialised_once_per_process(artifact_dir):
    _write_artifact(artifact_dir)

    assert registry.load_model() is registry.load_model()


def test_retraining_invalidates_the_cache(artifact_dir):
    """A new artifact on disk should be picked up without a restart."""
    _write_artifact(artifact_dir)
    first = registry.load_model()

    # Rewrite with a different size/mtime, which changes the cache key.
    _write_artifact(artifact_dir, pipeline=_tiny_pipeline())
    import os

    path = artifact_dir / "churn_model.joblib"
    os.utime(path, (0, 0))

    assert registry.load_model() is not first


def test_reset_forces_a_reload(artifact_dir):
    _write_artifact(artifact_dir)
    first = registry.load_model()
    registry.reset()

    assert registry.load_model() is not first
