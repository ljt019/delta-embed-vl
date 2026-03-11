# Delta Embed VL

Distills Qwen3-VL-Embedding-8B into Qwen3.5-0.8B-Base - a natively multimodal vision-language model using Gated DeltaNet + Gated Attention - to produce a sub-1B parameter embedder that handles text, images, and mixed-modal inputs in a single unified embedding space.

## Pipeline

```
prepare dataset → student distill → MTEB eval
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
  --max-length 2048 \
  --epochs 3
```

### Individual stages

```bash
uv run prepare-data \
  --teacher-device cuda:0 \
  --teacher-batch-size 8 \
  --max-length 2048
uv run train-model \
  --dataset-path data/prepared/balanced-all_student-qwen3-5-0-8b-base_teacher-qwen3-vl-embedding-8b_maxlen-2048 \
  --student-device cuda:1 \
  --batch-size 8
uv run eval-model
```

### Smoke test

```bash
uv run scripts/check_teacher.py --device cuda:0 --image-size 64
uv run scripts/check_student.py --device cuda:1 --image-size 64
uv run prepare-data \
  --limit 8 \
  --teacher-device cuda:0 \
  --teacher-batch-size 2 \
  --max-length 2048
uv run train-model \
  --dataset-path data/prepared/limit-8_student-qwen3-5-0-8b-base_teacher-qwen3-vl-embedding-8b_maxlen-2048 \
  --epochs 1 \
  --batch-size 2 \
  --max-length 2048 \
  --student-device cuda:1
```

## Project structure

```
src/delta_embed_vl/
  prepare_data.py  prepare-stage API + CLI
  train.py         train-stage logic + CLI
  eval.py          eval orchestration + CLI
  pipeline.py      end-to-end orchestration CLI
  data/            raw sources + teacher embed + dataset build
  model/           tokenization + embedding model loading + pooling
  evals/           concrete eval implementations
  settings.py      configuration
scripts/
  check_teacher.py   smoke test local teacher embeddings
  check_student.py   smoke test local student embeddings
```
