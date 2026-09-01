import re
import uuid
from pathlib import Path

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(original: str) -> str:
    """Generate a collision-free, path-traversal-safe filename that keeps the
    original extension (if any) for content-type sniffing on the frontend."""
    suffix = Path(original).suffix.lower()
    suffix = _SAFE_CHARS.sub("", suffix)[:10]
    return f"{uuid.uuid4().hex}{suffix}"


def dataset_dir(dataset_id: int) -> Path:
    from . import config

    d = config.DATASETS_DIR / str(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
