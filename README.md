# Delta Embed VL

Distills Qwen3-VL-Embedding-8B into Qwen3.5-0.8B-Base to produce a sub-1B parameter embedder that handles text, images, and mixed-modal inputs in a single unified embedding space.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

On Linux with CUDA, install accelerated attention:

```bash
uv sync --extra accel
```

## Configuration

All settings live in `config.toml`. Edit it, then run commands — no CLI flags needed.

## Usage

```bash
uv run prepare              # build dataset (normalize + teacher embed)
uv run prepare --push-to-hub  # build + upload to HuggingFace
uv run train                # train student model
uv run train --push-to-hub  # train + upload model to HuggingFace
uv run eval                 # run MTEB evaluation
```

If `data/dataset/` doesn't exist when you run `train`, it auto-downloads from the Hub dataset configured in `config.toml`.

## Project structure

```
config.toml              all configuration
src/delta_embed_vl/
  __init__.py            config loading, shared helpers
  prepare.py             dataset preparation entry point
  train.py               training loop
  eval.py                evaluation orchestrator
  data/
    build.py             dataset build pipeline (normalize → embed → save)
    sources.py           raw data → normalized samples
    teacher.py           teacher model loading + embedding
    download.py          raw data download + caching
  model/
    student.py           student model loading + projection head
    tokenization.py      shared multimodal tokenization
    pooling.py           last-token pooling + normalization
  evals/
    mteb_eval.py         MTEB evaluation implementation
scripts/
  check_teacher.py       smoke test teacher embeddings
  check_student.py       smoke test student embeddings
```
