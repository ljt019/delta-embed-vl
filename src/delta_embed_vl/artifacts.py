from __future__ import annotations

from delta_embed_vl.settings import Settings

_settings = Settings()


def versioned_name(base: str, *, limit: int | None = None) -> str:
    name = f"{base}_{_settings.artifact_version}"
    if limit is not None:
        name += f"_test{limit}"
    return name
