"""One-command setup: raw dataset -> warehouse -> RFM -> churn model -> scores.

    python scripts/seed_all.py

This is the entry point a reviewer runs after `docker-compose up db` (or with
the default SQLite URL, after nothing at all). It chains the per-stage scripts
that already exist and are individually runnable:

    run_etl.py           download + clean + load the star schema
    run_rfm.py           compute RFM scores and segments onto dim_customer
    train_churn_model.py fit logistic regression, write artifact + metadata
    score_customers.py   apply the model, persist churn_risk_score
    run_stats.py         hypothesis tests (printed; cheap, no persistence)

WHY SUBPROCESSES RATHER THAN IMPORTS
Each stage is a process boundary on purpose. The ETL stage holds the full
~1M-row frame in memory; running it in-process would keep pandas' peak
allocation alive for the rest of the run, and the training stage then loads its
own frame on top of that. Letting the OS reclaim each stage's memory on exit
keeps the ceiling at one stage's worth rather than the sum. It also means a
stage that dies from an unexpected exit (not just an exception) is still
reported here as a failed stage rather than taking this orchestrator down with
it.

The cost is that stages cannot share a transaction. That is fine: each stage is
individually idempotent, so a failure mid-run is recovered by re-running this
script, and `--from` lets you resume at the stage that failed instead of
repeating the ~5-minute ETL.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401  (path setup; must precede `app` imports)

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from app.core.config import get_settings

SCRIPTS_DIR = Path(__file__).resolve().parent


def _masked_url(url: str) -> str:
    """Render a DB URL with the password redacted, for safe console output.

    SQLAlchemy already knows how to do this correctly across every URL shape
    (including passwords containing '@' or ':'), so parsing it by hand would
    only introduce bugs. A malformed URL is reported as-is rather than raising:
    this line is a diagnostic banner, and the connection attempt that follows
    will produce the real, actionable error.
    """
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 - never fail a run over a banner
        return url



@dataclass(frozen=True)
class Stage:
    """A single pipeline step.

    `key` is what the user passes to --from/--only, `script` is the file to run,
    and `description` is shown in the banner so the console output explains
    itself while a long stage is running.
    """

    key: str
    script: str
    description: str
    args: tuple[str, ...] = ()


STAGES: tuple[Stage, ...] = (
    Stage("etl", "run_etl.py", "Download, clean and load the star schema"),
    Stage("rfm", "run_rfm.py", "Compute RFM scores and assign segments"),
    Stage("train", "train_churn_model.py", "Train the churn model and write artifacts"),
    Stage("score", "score_customers.py", "Score every customer's churn risk"),
    Stage("stats", "run_stats.py", "Run the hypothesis tests"),
)

STAGE_KEYS = [s.key for s in STAGES]


def _banner(text: str, char: str = "=") -> None:
    print(f"\n{char * 78}\n{text}\n{char * 78}", flush=True)


def _run_stage(stage: Stage, index: int, total: int) -> float:
    """Run one stage, streaming its output. Returns elapsed seconds.

    Raises SystemExit with the child's return code if the stage fails, so the
    orchestrator's exit status is usable in CI and by `make setup`.
    """
    _banner(f"[{index}/{total}] {stage.key} - {stage.description}")
    started = time.perf_counter()

    # sys.executable, not "python": guarantees the stage runs in the same
    # interpreter/virtualenv as this script, which is the usual cause of
    # "works when I run it directly, ImportError from the seed script".
    cmd = [sys.executable, str(SCRIPTS_DIR / stage.script), *stage.args]

    # No capture: the child writes straight to this terminal, so progress from
    # a multi-minute ETL is visible live instead of arriving in one dump at the
    # end (or never, if it hangs).
    completed = subprocess.run(cmd, cwd=str(SCRIPTS_DIR.parent), check=False)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        _banner(
            f"STAGE FAILED: {stage.key} (exit {completed.returncode})\n"
            f"Fix the error above, then resume without redoing earlier stages:\n"
            f"    python scripts/seed_all.py --from {stage.key}",
            char="!",
        )
        raise SystemExit(completed.returncode)

    print(f"\n-- {stage.key} finished in {elapsed:.1f}s", flush=True)
    return elapsed


def _select_stages(start_from: str | None, only: str | None) -> list[Stage]:
    if only:
        return [s for s in STAGES if s.key == only]
    if start_from:
        offset = STAGE_KEYS.index(start_from)
        return list(STAGES[offset:])
    return list(STAGES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full data + model pipeline end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/seed_all.py                # everything\n"
            "  python scripts/seed_all.py --from train   # resume after a failed train\n"
            "  python scripts/seed_all.py --only rfm     # re-run one stage\n"
        ),
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=STAGE_KEYS,
        help="Resume from this stage, skipping the ones before it.",
    )
    parser.add_argument(
        "--only",
        choices=STAGE_KEYS,
        help="Run exactly one stage.",
    )
    args = parser.parse_args()

    if args.start_from and args.only:
        parser.error("--from and --only are mutually exclusive")

    settings = get_settings()
    stages = _select_stages(args.start_from, args.only)

    _banner("Retail Insights Platform - full pipeline seed")
    # Echoing the target database up front prevents the classic confusion of
    # seeding SQLite while the API container reads Postgres. The URL is masked
    # because it can carry a password.
    print(f"Database   : {_masked_url(settings.effective_database_url)}")

    print(f"Stages     : {' -> '.join(s.key for s in stages)}")
    print(f"Interpreter: {sys.executable}")
    print(
        "\nFirst run downloads ~45MB and takes roughly 5-10 minutes end to end;\n"
        "the dataset is cached afterwards, so later runs are much faster.",
        flush=True,
    )

    timings: list[tuple[str, float]] = []
    overall_started = time.perf_counter()

    if stages[0].key == "etl":
        settings.ensure_directories()
        alembic_ini = SCRIPTS_DIR.parent / "backend" / "alembic.ini"
        command.upgrade(Config(str(alembic_ini)), "head")

    for index, stage in enumerate(stages, start=1):
        timings.append((stage.key, _run_stage(stage, index, len(stages))))

    total_elapsed = time.perf_counter() - overall_started

    _banner("PIPELINE COMPLETE")
    width = max(len(key) for key, _ in timings)
    for key, elapsed in timings:
        print(f"  {key.ljust(width)}  {elapsed:7.1f}s")
    print(f"  {'TOTAL'.ljust(width)}  {total_elapsed:7.1f}s")
    print(
        "\nNext:\n"
        "  1. Start the API:      uvicorn app.main:app --reload --app-dir backend\n"
        "  2. Start the frontend: npm run dev --prefix frontend\n"
        "  3. Open the docs:      http://localhost:8000/docs\n"
        "     Open the dashboard: http://localhost:5173\n",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
