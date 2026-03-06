from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from PIL import Image

from delta_embed_vl.settings import Settings

ImageLike = Image.Image | dict[str, Any] | str | Path | None

_RAW_DATA_DIR = Settings().data_dir / "raw"


def resolve_image_path(image_path: str | Path) -> Path | None:
    path = Path(image_path)
    if path.exists():
        return path

    marker = "/downloads/extracted/"
    normalized_path = str(image_path).replace("\\", "/")
    if marker in normalized_path:
        suffix = normalized_path.split(marker, maxsplit=1)[1]
        matches = _RAW_DATA_DIR.glob(f"**/downloads/extracted/{suffix}")
        resolved = next(matches, None)
        if resolved is not None and resolved.exists():
            return resolved

    return None


def coerce_image_to_rgb(image: ImageLike) -> Image.Image | None:
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        resolved_path = resolve_image_path(image)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
            return loaded.convert("RGB")

    image_bytes = image.get("bytes")
    if image_bytes is not None:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        elif isinstance(image_bytes, bytearray):
            image_bytes = bytes(image_bytes)
        elif isinstance(image_bytes, list):
            image_bytes = bytes(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as loaded:
            return loaded.convert("RGB")

    image_path = image.get("path")
    if image_path:
        resolved_path = resolve_image_path(image_path)
        if resolved_path is None:
            return None
        with Image.open(resolved_path) as loaded:
            return loaded.convert("RGB")

    return None


def has_usable_image(image: ImageLike) -> bool:
    if image is None:
        return False
    if isinstance(image, Image.Image):
        return True
    if isinstance(image, (str, Path)):
        return resolve_image_path(image) is not None

    if image.get("bytes") is not None:
        return True

    image_path = image.get("path")
    return bool(image_path) and resolve_image_path(image_path) is not None


def image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
