"""Lazy, fail-soft loader for the churn model artifact.

Why this module exists
----------------------
A trained model is a *file*, and files go missing. A reviewer who clones this
repo and starts the API before running ``scripts/train_churn_model.py`` has no
``churn_model.joblib`` on disk. The naive approach - ``joblib.load()`` at import
time - means the whole server refuses to boot, so the executive dashboard, the
segment charts and the health check all die because one optional capability is
absent.

Instead:

* Loading is **lazy**: nothing touches the filesystem until an endpoint actually
  needs the model.
* Loading is **cached**: the artifact is deserialised once per process, not once
  per request. The cache is keyed on the file's mtime+size, so retraining is
  picked up without a restart.
* Failure is **typed**: a missing, unreadable, or structurally wrong artifact
  raises :class:`ModelNotLoadedError`, which the global handler renders as a 503
  with an actionable message. Model-backed endpoints degrade; everything else
  keeps serving.

Corruption is treated as a first-class case, not just absence. A truncated
joblib file, a pickle written by an incompatible scikit-learn, or an artifact
whose feature order no longer matches :data:`FEATURE_NAMES` all raise the same
typed error rather than surfacing as a 500 or - worse - silently mis-scoring
every customer because the coefficients are positional.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.core.exceptions import ModelNotLoadedError
from app.core.logging_config import get_logger
from app.ml.features import FEATURE_NAMES

logger = get_logger(__name__)

# Guards the module-level cache. Uvicorn serves requests from a thread pool, so
# two concurrent first-requests could otherwise both deserialise the artifact.
_lock = threading.Lock()


@dataclass(frozen=True)
class _CacheKey:
    """Identity of the artifact pair on disk.

    Comparing mtime and size lets a retrained model be picked up automatically:
    the key changes, so the cache is invalidated. Cheaper and less surprising
    than hashing several MB of pickle on every request.
    """

    model_mtime: float
    model_size: int
    metadata_mtime: float


@dataclass
class LoadedModel:
    """A deserialised pipeline plus the model card that explains it."""

    pipeline: Any
    metadata: dict[str, Any]
    model_path: Path
    metadata_path: Path
    features: list[str]

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))



_cache: LoadedModel | None = None
_cache_key: _CacheKey | None = None


def _stat_or_missing(path: Path, label: str) -> Any:
    """Stat a required artifact, raising a typed error when it is absent."""
    if not path.exists():
        raise ModelNotLoadedError(
            f"The {label} is missing at {path}. Run "
            "`python scripts/train_churn_model.py` to train and persist it.",
            details={"missing_file": str(path), "artifact": label},
        )
    stat = path.stat()
    if stat.st_size == 0:
        raise ModelNotLoadedError(
            f"The {label} at {path} is empty, which means a previous training "
            "run was interrupted. Re-run `python scripts/train_churn_model.py`.",
            details={"corrupt_file": str(path), "reason": "zero-length file"},
        )
    return stat


def _unwrap(raw: Any, model_path: Path) -> tuple[Any, list[str]]:
    """Extract the estimator and its feature order from the saved artifact.

    ``train.py`` persists a bundle rather than a bare estimator::

        {"pipeline": Pipeline, "feature_names": [...],
         "model_version": "1.0.0", "trained_at": "..."}

    Bundling the feature order *with* the weights is deliberate: it makes the
    artifact self-describing, so the positional contract travels with the file
    instead of living only in a JSON sidecar that could drift. A bare estimator
    is still accepted for forward compatibility.
    """
    if isinstance(raw, dict):
        pipeline = raw.get("pipeline")
        if pipeline is None:
            raise ModelNotLoadedError(
                "The model bundle has no 'pipeline' key, so it was not written "
                f"by this project's training script (keys: {sorted(raw)[:6]}). "
                "Re-run `python scripts/train_churn_model.py`.",
                details={"artifact": str(model_path), "bundle_keys": sorted(map(str, raw))},
            )
        return pipeline, list(raw.get("feature_names") or [])
    return raw, []


def _validate(
    pipeline: Any,
    bundle_features: list[str],
    metadata: dict[str, Any],
    model_path: Path,
) -> list[str]:
    """Reject artifacts that load but cannot be trusted, and fix the feature order.

    Three checks matter:

    1. The estimator must actually be able to produce probabilities. A pickle of
       the wrong object (or one saved by an incompatible sklearn) will load
       happily and then fail at call time, deep inside a request.
    2. The bundle's feature order and the model card's must agree with each
       other - disagreement means the two files came from different runs.
    3. Both must match :data:`FEATURE_NAMES`. The scaler and coefficients are
       positional, so a stale artifact trained on a different feature order
       would return confident nonsense. Silent wrongness is far worse than a
       503, so it is treated as corruption.

    Returns:
        The authoritative feature order to score with.
    """
    if not hasattr(pipeline, "predict_proba"):
        raise ModelNotLoadedError(
            "The loaded artifact is not a probabilistic classifier "
            f"(got {type(pipeline).__name__}, which has no predict_proba). "
            "The model file is likely corrupt - please retrain.",
            details={"artifact": str(model_path), "loaded_type": type(pipeline).__name__},
        )

    card_features = list(metadata.get("features") or [])
    if bundle_features and card_features and bundle_features != card_features:
        raise ModelNotLoadedError(
            "The model artifact and its metadata file disagree about the "
            "feature order, so they were written by different training runs. "
            "Re-run `python scripts/train_churn_model.py` to regenerate both.",
            details={
                "artifact_features": bundle_features,
                "metadata_features": card_features,
            },
        )

    resolved = bundle_features or card_features or list(FEATURE_NAMES)
    if resolved != FEATURE_NAMES:
        raise ModelNotLoadedError(
            "The stored model was trained on a different feature set than the "
            "current code expects, so its coefficients no longer line up. "
            "Re-run `python scripts/train_churn_model.py`.",
            details={
                "artifact_features": resolved,
                "expected_features": list(FEATURE_NAMES),
            },
        )
    return resolved



def load_model(*, force: bool = False) -> LoadedModel:
    """Return the cached model, loading it from disk on first use.

    Raises:
        ModelNotLoadedError: if either artifact is missing, empty, unreadable,
            or structurally inconsistent with the current feature contract.
    """
    global _cache, _cache_key

    settings = get_settings()
    model_path = settings.model_file
    metadata_path = settings.model_metadata_file

    model_stat = _stat_or_missing(model_path, "churn model artifact")
    metadata_stat = _stat_or_missing(metadata_path, "model metadata file")

    key = _CacheKey(
        model_mtime=model_stat.st_mtime,
        model_size=model_stat.st_size,
        metadata_mtime=metadata_stat.st_mtime,
    )

    with _lock:
        if not force and _cache is not None and _cache_key == key:
            return _cache

        # Imported here rather than at module scope: joblib pulls in a good deal
        # of scipy, and an API process that never touches the model should not
        # pay that import cost at startup.
        import joblib

        try:
            raw_artifact = joblib.load(model_path)
        except ModuleNotFoundError as exc:  # pickled against a different env

            raise ModelNotLoadedError(
                "The model artifact references a module that is not installed "
                f"({exc.name}). It was probably trained in a different "
                "environment - retrain it here to fix the mismatch.",
                details={"artifact": str(model_path), "missing_module": exc.name},
            ) from exc
        except Exception as exc:  # truncated / non-pickle / version mismatch
            raise ModelNotLoadedError(
                f"The churn model at {model_path} could not be deserialised "
                f"({type(exc).__name__}). The file appears corrupt - re-run "
                "`python scripts/train_churn_model.py`.",
                details={"artifact": str(model_path), "reason": str(exc)[:300]},
            ) from exc

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelNotLoadedError(
                f"The model metadata at {metadata_path} is not valid JSON "
                f"(line {exc.lineno}). Re-run the training script.",
                details={"artifact": str(metadata_path), "reason": exc.msg},
            ) from exc

        if not isinstance(metadata, dict):
            raise ModelNotLoadedError(
                "The model metadata file does not contain a JSON object.",
                details={"artifact": str(metadata_path)},
            )

        pipeline, bundle_features = _unwrap(raw_artifact, model_path)
        features = _validate(pipeline, bundle_features, metadata, model_path)

        _cache = LoadedModel(
            pipeline=pipeline,
            metadata=metadata,
            model_path=model_path,
            metadata_path=metadata_path,
            features=features,
        )

        _cache_key = key
        logger.info(
            "Loaded churn model v%s (trained %s, ROC-AUC %s)",
            _cache.version,
            metadata.get("trained_at", "unknown"),
            metadata.get("metrics", {}).get("roc_auc", "n/a"),
        )
        return _cache


def get_metadata() -> dict[str, Any]:
    """Return the model card, loading the artifact if needed."""
    return load_model().metadata


def is_available() -> bool:
    """Whether the model can be served right now.

    Used by ``/api/health`` to report degraded capability without turning a
    missing model into a failed health check. Never raises.
    """
    try:
        load_model()
        return True
    except ModelNotLoadedError:
        return False
    except Exception:  # pragma: no cover - defensive: health must never 500
        logger.exception("Unexpected error while probing model availability")
        return False


def reset() -> None:
    """Drop the cache so the next call reloads from disk.

    Exists for tests, which swap the artifact directory between cases, and for
    the (rare) operational need to hot-reload a retrained model.
    """
    global _cache, _cache_key
    with _lock:
        _cache = None
        _cache_key = None


def predict_proba(features: pd.DataFrame | dict[str, float]) -> np.ndarray:
    """Score one or many customers, returning P(churn) in ``[0, 1]``.

    Accepts a mapping for a single customer or a frame for a batch. Columns are
    reordered to the trained feature order before scoring - callers should not
    have to know that the pipeline is positional, and relying on dict ordering
    here would be a silent-wrongness bug waiting to happen.

    Raises:
        ModelNotLoadedError: if the artifact is unavailable.
        InsufficientDataError: never - missing features are a caller error and
            surface as a ``ModelNotLoadedError``-free ``ValueError`` upstream,
            so validation belongs in the Pydantic layer.
    """
    model = load_model()
    frame = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()

    missing = [name for name in model.features if name not in frame.columns]
    if missing:
        # Reaching here means a caller bypassed the request schema; be explicit
        # rather than letting sklearn raise a shape error.
        raise ModelNotLoadedError(
            "Cannot score: the request is missing features the model requires.",
            details={"missing_features": missing, "expected": model.features},
        )

    ordered = frame[model.features].astype(float)
    probabilities = model.pipeline.predict_proba(ordered)[:, 1]
    return np.asarray(probabilities, dtype=float)


__all__ = [
    "LoadedModel",
    "load_model",
    "get_metadata",
    "is_available",
    "reset",
    "predict_proba",
]
