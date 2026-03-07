# Delta Embed VL

Distills Qwen3-VL-Embedding-8B into Qwen3.5-0.8B-Base - a natively multimodal vision-language model using Gated DeltaNet + Gated Attention - to produce a sub-1B parameter embedder that handles text, images, and mixed-modal inputs in a single unified embedding space.

## Pipeline

```
download → preprocess → local teacher/student distill → MTEB eval
```

**Data sources:** Wikipedia (text) + [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) (30 multimodal VQA/captioning/OCR/chart/table configs).

**Teacher:** Qwen3-VL-Embedding-8B loaded in-process as a frozen local model.

**Student:** Qwen3.5-0.8B-Base with last-token pooling + L2 normalization, trained to match teacher embeddings with cosine loss.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### Run the full pipeline

```bash
uv run delta-embed-pipeline \
  --teacher-device cuda:0 \
  --student-device cuda:1 \
  --teacher-batch-size 8 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --max-length 512 \
  --epochs 3
```

### Individual stages

```bash
uv run prepare-data
uv run train-model \
  --teacher-device cuda:0 \
  --student-device cuda:1 \
  --teacher-batch-size 8 \
  --batch-size 8
uv run eval-model
```

### Smoke test

```bash
uv run scripts/check_teacher.py --device cuda:0 --image-size 64
uv run train-model \
  --limit 8 \
  --epochs 1 \
  --batch-size 2 \
  --teacher-batch-size 2 \
  --teacher-device cuda:0 \
  --student-device cuda:1
```

## Project structure

```
src/delta_embed_vl/
  data/        download, preprocess, dataloader
  teacher/     teacher length audit utilities
  model/       teacher + student model loading, pooling
  training/    local teacher-student distillation loop
  eval/        MTEB evaluation wrapper
  main.py      CLI entrypoints
  settings.py  configuration (data_dir, teacher model, token limits)
scripts/
  check_teacher.py   smoke test local teacher embeddings
```
