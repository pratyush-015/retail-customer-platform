"""Data-quality auditing for the ETL pipeline.

The requirement driving this module: **no row is ever dropped or modified
silently.** Every cleaning decision is recorded with

* a stable ``rule`` identifier,
* how many rows it affected (and what share of the input that was),
* the action taken (``dropped`` / ``modified`` / ``flagged`` / ``retained``),
* a human-readable rationale,
* a few example rows, so a reviewer can spot-check the decision rather than
  trusting the count.

The accumulated record is written to ``data/data_quality_report.json``, which
turns the ETL from a black box into something auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.core.logging_config import get_logger

logger = get_logger(__name__)

Action = Literal["dropped", "modified", "flagged", "retained"]


def _json_safe(value: Any) -> Any:
    """Coerce numpy/pandas/datetime scalars into JSON-serialisable values."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass
class CleaningDecision:
    """A single auditable cleaning action."""

    rule: str
    action: Action
    rows_affected: int
    reason: str
    pct_of_input: float = 0.0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "action": self.action,
            "rows_affected": self.rows_affected,
            "pct_of_input": round(self.pct_of_input, 4),
            "reason": self.reason,
            "examples": [_json_safe(e) for e in self.examples],
        }


@dataclass
class DataQualityReport:
    """Accumulates cleaning decisions and renders the audit artifact."""

    source_file: str = ""
    rows_in: int = 0
    rows_out: int = 0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str | None = None
    decisions: list[CleaningDecision] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ log --
    def record(
        self,
        rule: str,
        action: Action,
        rows_affected: int,
        reason: str,
        examples: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record one cleaning decision and echo it to the log."""
        pct = (rows_affected / self.rows_in * 100) if self.rows_in else 0.0
        decision = CleaningDecision(
            rule=rule,
            action=action,
            rows_affected=int(rows_affected),
            reason=reason,
            pct_of_input=pct,
            examples=examples or [],
        )
        self.decisions.append(decision)

        # Zero-impact rules are still recorded (proving the check ran) but do
        # not deserve a prominent log line.
        level = logger.info if rows_affected else logger.debug
        level(
            "  [%-8s] %-32s %8s rows (%5.2f%%)  %s",
            action.upper(),
            rule,
            f"{rows_affected:,}",
            pct,
            reason,
        )

    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = _json_safe(value)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning("  %s", message)

    # --------------------------------------------------------------- derive --
    @property
    def rows_dropped(self) -> int:
        return sum(d.rows_affected for d in self.decisions if d.action == "dropped")

    @property
    def rows_modified(self) -> int:
        return sum(d.rows_affected for d in self.decisions if d.action == "modified")

    @property
    def retention_rate(self) -> float:
        return (self.rows_out / self.rows_in * 100) if self.rows_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": self.source_file,
            "summary": {
                "rows_in": self.rows_in,
                "rows_out": self.rows_out,
                "rows_dropped": self.rows_dropped,
                "rows_modified": self.rows_modified,
                "retention_rate_pct": round(self.retention_rate, 2),
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            },
            "decisions": [d.to_dict() for d in self.decisions],
            "metrics": self.metrics,
            "warnings": self.warnings,
        }

    # ---------------------------------------------------------------- write --
    def save(self, path: Path) -> Path:
        """Write the report atomically so a crash never leaves it truncated."""
        self.finished_at = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        tmp.replace(path)

        logger.info("Data quality report written to %s", path)
        return path

    def log_summary(self) -> None:
        """Print the headline numbers at the end of a run."""
        logger.info("-" * 74)
        logger.info(
            "Cleaning summary: %s rows in -> %s rows out "
            "(%s dropped, %s modified, %.2f%% retained)",
            f"{self.rows_in:,}",
            f"{self.rows_out:,}",
            f"{self.rows_dropped:,}",
            f"{self.rows_modified:,}",
            self.retention_rate,
        )
        if self.warnings:
            logger.info("%d warning(s) recorded - see the report.", len(self.warnings))
        logger.info("-" * 74)


__all__ = ["DataQualityReport", "CleaningDecision"]
