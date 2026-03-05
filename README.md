# Delta Embed VL

Distills Qwen3-VL-Embedding-8B into Qwen3.5-0.8B-Base - a natively multimodal vision-language model using Gated DeltaNet + Gated Attention - to produce a sub-1B parameter embedder that handles text, images, and mixed-modal inputs in a single unified embedding space.

## Pipeline

```
download → preprocess → teacher embeddings → distill → MTEB eval
```

**Data sources:** Wikipedia (text) + [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) (30 multimodal VQA/captioning/OCR/chart/table configs).

**Teacher:** Qwen3-VL-Embedding-8B served locally via vLLM, generates target embeddings over the full dataset.

**Student:** Qwen3.5-0.8B-Base with last-token pooling + L2 normalization, trained to match teacher embeddings with cosine loss.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Start the teacher server

```bash
uv run scripts/start_teacher.py
```

### Run the full pipeline

```bash
uv run delta-embed-pipeline \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --max-length 512 \
  --epochs 3
```

### Individual stages

```bash
uv run prepare-data                # download + preprocess + teacher embeddings
uv run train-model --batch-size 8  # distillation only
uv run eval-model                  # MTEB eval from checkpoints/
```

### Smoke test

```bash
uv run delta-embed-pipeline --limit 10 --batch-size 4 --grad-accum-steps 8
```

## Project structure

```
src/delta_embed/
  data/        download, preprocess, dataloader
  teacher/     teacher embedding generation (async, vLLM API)
  model/       student model loading, pooling
  training/    cosine distillation loop
  eval/        MTEB evaluation wrapper
  main.py      CLI entrypoints
  settings.py  configuration (data_dir, teacher URL, concurrency)
scripts/
  start_teacher.py   launch vLLM teacher server
  test_pipeline.py   quick pipeline sanity check
  check_teacher.py   verify teacher server is responding
```
