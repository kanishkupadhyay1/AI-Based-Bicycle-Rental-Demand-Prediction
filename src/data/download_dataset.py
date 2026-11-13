"""Download and extract the official UCI Bike Sharing Dataset."""

import io
import time
import zipfile
from pathlib import Path

import requests

from src.config import (
    RAW_DATA_DIR,
    RAW_DAY_CSV,
    RAW_HOUR_CSV,
    RAW_README_TXT,
    UCI_DATASET_ZIP_URL,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)


def download_uci_bike_sharing_dataset(
    url: str = UCI_DATASET_ZIP_URL,
    output_dir: Path = RAW_DATA_DIR,
    max_retries: int = 3,
    timeout: int = 60,
    force: bool = False,
) -> Path:
    """Download and extract the UCI Bike Sharing Dataset zip archive.

    Args:
        url: Official UCI dataset archive URL.
        output_dir: Directory to extract raw files into.
        max_retries: Number of download attempts before failing.
        timeout: HTTP request timeout in seconds.
        force: If True, re-download even if files already exist.

    Returns:
        Path to the directory containing extracted dataset files.

    Raises:
        RuntimeError: If download or archive extraction fails after retries.
    """
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [RAW_HOUR_CSV, RAW_DAY_CSV, RAW_README_TXT]
    if not force and all(f.exists() and f.stat().st_size > 0 for f in required_files):
        logger.info("Official UCI Bike Sharing Dataset already present in %s", output_dir)
        return output_dir

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading UCI Bike Sharing Dataset (attempt %d/%d) from %s",
                attempt,
                max_retries,
                url,
            )
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            zip_bytes = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_bytes, "r") as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"Corrupted member in zip archive: {bad_file}")
                zf.extractall(output_dir)

            # Save archive locally for reproducibility
            archive_path = output_dir / "bike+sharing+dataset.zip"
            archive_path.write_bytes(response.content)

            for req in required_files:
                if not req.exists() or req.stat().st_size == 0:
                    raise FileNotFoundError(f"Expected file missing after extraction: {req}")

            logger.info("Successfully downloaded and extracted UCI dataset to %s", output_dir)
            return output_dir

        except (requests.RequestException, zipfile.BadZipFile, FileNotFoundError, OSError) as exc:
            last_error = exc
            logger.warning("Download attempt %d failed: %s", attempt, exc)
            if attempt < max_retries:
                wait_seconds = 2 ** attempt
                logger.info("Retrying in %d seconds...", wait_seconds)
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to download official UCI Bike Sharing Dataset after {max_retries} attempts: {last_error}"
    )


if __name__ == "__main__":
    download_uci_bike_sharing_dataset()
