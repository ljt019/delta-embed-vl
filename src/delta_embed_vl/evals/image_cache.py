from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image

logger = logging.getLogger(__name__)
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def load_remote_image(url: str, *, cache_dir: Path) -> Image.Image:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(url).path).suffix or ".jpg"
    cache_path = cache_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}{suffix}"
    if not cache_path.exists():
        tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        with httpx.stream(
            "GET",
            url,
            follow_redirects=True,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as file:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    file.write(chunk)
        tmp_path.replace(cache_path)
        logger.debug("Cached image asset: %s", cache_path)

    with Image.open(cache_path) as image:
        return image.convert("RGB")
