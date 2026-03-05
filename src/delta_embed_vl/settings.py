from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("data")

    teacher_base_url: str = "http://localhost:8000/v1"
    teacher_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    teacher_concurrency: int = 20
