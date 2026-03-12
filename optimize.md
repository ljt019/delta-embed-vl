# optimize.md

Autonomous optimization of the data preparation pipeline, focused on teacher embedding throughput.

## Goal

Minimize wall-clock time for `uv run prepare --limit 50 --rebuild-normalized --detailed-timings` while producing the exact same output rows. Lower `prepare_total` elapsed_s is better. Do not use `--no-stream` — raw data is cached after the first run automatically.

**Priority**: Teacher embedding is ~85% of total prep time. Focus optimization efforts there. Normalization is already well-optimized from the previous research round.

## Setup

Work with the user to:

1. **Agree on a run tag** based on today's date (e.g. `mar12`). Branch `autoresearch/prep-<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/prep-<tag>`
3. **Read the mutable files** for full context:
   - `src/delta_embed_vl/data/build.py` — normalization orchestration, CPU parallelism, teacher embedding, dataset assembly.
   - `src/delta_embed_vl/data/sources.py` — per-source normalization logic, text chunking, image processing, tokenizer fitting.
   - `src/delta_embed_vl/data/teacher.py` — teacher model loading and embedding inference.
4. **Read the fixed files** for context only (do not modify):
   - `src/delta_embed_vl/data/download.py` — raw data caching.
   - `src/delta_embed_vl/model/tokenization.py` — tokenizer and processor utilities.
   - `src/delta_embed_vl/prepare.py` — CLI entry point.
   - `config.toml` — project configuration.
5. **Verify raw cache is warm**: Run `ls data/raw/` and confirm wikipedia and cauldron directories exist. If not, tell the human to run `uv run prepare --limit 50` once first.
6. **Initialize prep_results.tsv** with just the header row.
7. **Confirm and go**.

## The benchmark command

Always run exactly this:

```bash
uv run prepare --limit 50 --rebuild-normalized --detailed-timings > run.log 2>&1
```

This ensures:
- Raw data comes from local cache (no network noise).
- Normalized cache is always rebuilt (tests your code changes).
- Teacher embedding always re-runs.
- Detailed timing lines are emitted for comparison.

## What you CAN modify

- `src/delta_embed_vl/data/build.py`
- `src/delta_embed_vl/data/sources.py`
- `src/delta_embed_vl/data/teacher.py`

Everything in these files is fair game: parallelism strategy, batch sizes, data structures, Arrow writing, image processing, tokenizer usage, teacher inference, memory layout, etc.

## What you CANNOT modify

- `src/delta_embed_vl/data/download.py`
- `src/delta_embed_vl/prepare.py`
- `src/delta_embed_vl/train.py`
- `src/delta_embed_vl/eval.py`
- `src/delta_embed_vl/__init__.py`
- `src/delta_embed_vl/model/` (except indirectly via teacher.py)
- `config.toml`
- `pyproject.toml`
- Do not install new packages or add dependencies.

## Reading results

After a run, extract the key metrics:

```bash
grep "TIMING stage=normalize_total\|TIMING stage=embed_total\|TIMING stage=embed_compute\|TIMING stage=embed_teacher_load\|TIMING stage=prepare_total" run.log
grep "Dataset saved: rows=" run.log
```

**Primary metric**: `prepare_total` elapsed_s — lower is better.

**Key secondary metrics** (focus your analysis here):
- `embed_total` — total GPU teacher embedding time (includes model load). This is the dominant cost.
- `embed_teacher_load` — teacher model load time. One-time cost per run.
- `embed_compute` — pure embedding inference time. The main target for optimization.
- `embed_shard` — per-device shard timings (rows_per_s is the key throughput number).

**Lower priority** (already well-optimized):
- `normalize_total` — CPU normalization time.
- `normalize_generate` — normalization worker time.

## Correctness rule

The final `Dataset saved: rows=N` **must match the baseline row count exactly**. If row count changes, the run is invalid — discard it even if faster. The pipeline must produce the same data, just faster.

If a run crashes, check `tail -n 50 run.log` for the traceback. Fix if trivial, otherwise discard and move on.

## Logging results

Log results to `prep_results.tsv` (tab-separated, untracked by git).

Header and columns:

```
commit	prepare_s	normalize_s	embed_s	rows	status	description
```

- `commit`: short git hash (7 chars)
- `prepare_s`: prepare_total elapsed_s
- `normalize_s`: normalize_total elapsed_s
- `embed_s`: embed_total elapsed_s
- `rows`: final row count
- `status`: `keep`, `discard`, or `crash`
- `description`: short text of what the experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at the current git state and results so far.
2. Edit one or more of the allowed files with an optimization idea.
3. `git commit -m "description of change"`
4. Run: `uv run prepare --limit 50 --rebuild-normalized --detailed-timings > run.log 2>&1`
5. Extract results: `grep "TIMING stage=\(normalize_total\|embed_total\|prepare_total\)" run.log` and `grep "Dataset saved: rows=" run.log`
6. If grep output is empty, the run crashed. `tail -n 50 run.log` to diagnose.
7. Record results in `prep_results.tsv`.
8. **If `prepare_total` improved AND row count matches baseline**: keep the commit, advance the branch.
9. **If `prepare_total` is equal or worse, OR row count changed**: `git reset --hard HEAD~1` to discard.

## Embedding optimization ideas to prioritize

Since embedding is ~85% of total time, focus here:

- `torch.compile` on the teacher model forward pass.
- Teacher model dtype experiments (bf16 vs fp16 vs mixed).
- Batching strategy: dynamic batch sizing based on sequence length.
- Reducing CPU-GPU data transfer overhead.
- Teacher processor optimizations (faster tokenization, image preprocessing).
- Pinned memory for CPU→GPU transfers.
- Reducing Arrow serialization overhead in the embedding write path.
- Async overlap: start next batch preprocessing while current batch is on GPU.
- CUDA graph capture for repeated inference patterns.
- Reducing teacher model load time (lazy loading, memory mapping).

Lower priority (normalization is already fast):
- Normalization buffer sizes.
- Image processing tweaks.
- Process pool tuning.

## Important rules

- **NEVER STOP**. Once the loop begins, do not pause to ask the human. They may be away. Run indefinitely until manually interrupted.
- **Do not modify frozen files**. Ever.
- **Do not change the benchmark command**. The command is the ground truth.
- **Row count is sacred**. Same rows or discard.
- **Simplicity matters**. A tiny speedup that adds ugly complexity is not worth it. Removing code for equal speed is a win.
