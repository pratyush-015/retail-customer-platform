"""Dataset acquisition with local caching.

The "Online Retail II" dataset (UCI ML Repository, ~45 MB) is downloaded once
and cached under ``data/raw/``. Re-running the pipeline reuses the cache, so a
reviewer pays the download cost exactly once and can work offline afterwards.

The cached file is validated before use - a truncated or HTML-error-page
download is detected here rather than surfacing as a baffling parse error 200
lines into the ETL.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests

from app.core.config import settings
from app.core.exceptions import DataSourceError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Official UCI mirror. The archive contains a single .xlsx with two sheets
# ("Year 2009-2010" and "Year 2010-2011").
DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ARCHIVE_NAME = "online_retail_II.zip"
EXCEL_NAME = "online_retail_II.xlsx"

# The real file is ~45 MB; anything much smaller is an error page or a partial
# transfer masquerading as success.
MIN_PLAUSIBLE_BYTES = 5 * 1024 * 1024


def _is_valid_excel(path: Path) -> bool:
    """Cheaply verify the cached file is a real, non-truncated .xlsx.

    An .xlsx is a zip archive, so an integrity check catches truncation without
    the cost of actually parsing a million rows.
    """
    if not path.exists() or path.stat().st_size < MIN_PLAUSIBLE_BYTES:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_dataset(
    dest_dir: Path | None = None,
    *,
    force: bool = False,
    timeout: int = 120,
) -> Path:
    """Ensure the dataset exists locally and return the path to the .xlsx.

    Parameters
    ----------
    dest_dir:
        Directory to cache into. Defaults to ``settings.raw_data_path``.
    force:
        Re-download even when a valid cached copy exists.

    Raises
    ------
    DataSourceError
        If the dataset cannot be downloaded or the result is not a valid
        workbook.
    """
    dest_dir = dest_dir or settings.raw_data_path
    dest_dir.mkdir(parents=True, exist_ok=True)

    excel_path = dest_dir / EXCEL_NAME
    archive_path = dest_dir / ARCHIVE_NAME

    if not force and _is_valid_excel(excel_path):
        size_mb = excel_path.stat().st_size / 1024 / 1024
        logger.info("Using cached dataset: %s (%.1f MB)", excel_path.name, size_mb)
        return excel_path

    # A previously-extracted archive is enough; no need to hit the network.
    if not force and archive_path.exists() and archive_path.stat().st_size >= MIN_PLAUSIBLE_BYTES:
        logger.info("Found cached archive, extracting: %s", archive_path.name)
        extracted = _extract_archive(archive_path, dest_dir)
        if _is_valid_excel(extracted):
            return extracted
        logger.warning("Cached archive did not yield a valid workbook; re-downloading.")

    logger.info("Downloading Online Retail II dataset from UCI (~45 MB)...")
    logger.info("  %s", DATASET_URL)

    try:
        with requests.get(DATASET_URL, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            total = int(response.headers.get("Content-Length", 0))
            written = 0
            last_logged_pct = -10

            tmp_path = archive_path.with_suffix(".part")
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = int(written * 100 / total)
                        if pct >= last_logged_pct + 10:
                            logger.info(
                                "  ... %3d%% (%.1f / %.1f MB)",
                                pct,
                                written / 1024 / 1024,
                                total / 1024 / 1024,
                            )
                            last_logged_pct = pct

            # Atomic rename: a half-written file is never left looking cached.
            tmp_path.replace(archive_path)

    except requests.RequestException as exc:
        raise DataSourceError(
            "Failed to download the Online Retail II dataset. Check your "
            "network connection, or download it manually and place it at "
            f"{excel_path}.",
            details={"url": DATASET_URL, "reason": str(exc)[:200]},
        ) from exc

    if archive_path.stat().st_size < MIN_PLAUSIBLE_BYTES:
        raise DataSourceError(
            "Downloaded file is implausibly small - the server likely returned "
            "an error page instead of the dataset.",
            details={
                "url": DATASET_URL,
                "bytes": archive_path.stat().st_size,
            },
        )

    extracted = _extract_archive(archive_path, dest_dir)
    if not _is_valid_excel(extracted):
        raise DataSourceError(
            "The downloaded dataset is not a readable .xlsx workbook.",
            details={"path": str(extracted)},
        )

    logger.info(
        "Dataset ready: %s (%.1f MB)",
        extracted.name,
        extracted.stat().st_size / 1024 / 1024,
    )
    return extracted


def _extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Extract the workbook from the downloaded zip.

    The archive's internal filename has varied across UCI revisions, so the
    workbook is located by extension rather than by an exact hardcoded name.
    """
    try:
        with zipfile.ZipFile(archive_path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".csv"))]
            if not members:
                raise DataSourceError(
                    "The dataset archive contains no .xlsx or .csv file.",
                    details={"archive": str(archive_path), "contents": zf.namelist()[:20]},
                )

            member = members[0]
            target = dest_dir / EXCEL_NAME
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            logger.debug("Extracted %s -> %s", member, target.name)
            return target

    except zipfile.BadZipFile as exc:
        raise DataSourceError(
            "The downloaded dataset archive is corrupted. Delete "
            f"{archive_path} and retry.",
            details={"archive": str(archive_path)},
        ) from exc


def resolve_dataset_path(explicit: str | Path | None = None) -> Path:
    """Resolve the dataset path, downloading only if necessary.

    An explicit path is honoured as-is (and must exist) so tests and offline
    runs can point at a fixture without touching the network.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise DataSourceError(
                f"The dataset file {path} does not exist.",
                details={"path": str(path)},
            )
        return path
    return download_dataset()


__all__ = ["download_dataset", "resolve_dataset_path", "DATASET_URL", "EXCEL_NAME"]
