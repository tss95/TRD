"""Local storage checks for feature extraction."""

from pathlib import Path
import shutil


def require_disk_budget(directory: Path, bytes_needed: int) -> None:
    """Reject an extraction before allocating a cache larger than free storage."""
    directory.mkdir(parents=True, exist_ok=True)
    if bytes_needed < 0 or shutil.disk_usage(directory).free < bytes_needed:
        raise OSError(f'Insufficient space for a {bytes_needed}-byte feature cache in {directory}')
