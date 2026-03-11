from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("data")

    teacher_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    student_model: str = "Qwen/Qwen3.5-0.8B-Base"
    hub_dataset: str = "ljt019/delta-embed-vl-dataset"
    wandb_project: str | None = "delta-embed-vl"
    seed: int = 42
    student_max_length: int = 2048
    teacher_max_model_len: int = 8192
    teacher_prompt_margin_tokens: int = 256

    wikipedia_ratio: float = 0.4
