from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("data")

    teacher_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    teacher_max_model_len: int = 8192
    teacher_prompt_margin_tokens: int = 256

    artifact_version: str = "qwen3_vl_contract_v2"
