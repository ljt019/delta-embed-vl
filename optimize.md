# prep_program.md

Autonomous optimization of the data preparation pipeline.

## Goal

Minimize wall-clock time for `uv run prepare --limit 25 --rebuild-normalized --detailed-timings` while producing the exact same output rows. Lower `prepare_total` elapsed_s is better.

## Setup

Work with the user to:

1. **Agree on a run tag** based on today's date (e.g. `mar11`). Branch `autoresearch/prep-<tag>` must not already exist.
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
5. **Verify raw cache is warm**: Run `ls data/raw/` and confirm wikipedia and cauldron directories exist. If not, tell the human to run `uv run prepare --limit 25` once first.
6. **Confirm and go**.

## The benchmark command

Always run exactly this:

```bash
uv run prepare --limit 25 --rebuild-normalized --detailed-timings > run.log 2>&1
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
grep "TIMING stage=normalize_total\|TIMING stage=embed_total\|TIMING stage=prepare_total" run.log
grep "Dataset saved: rows=" run.log
```

Example output:

```
TIMING stage=normalize_total elapsed_s=8.123 rows=1700 rows_per_s=209.3 limit=50
TIMING stage=embed_total elapsed_s=25.456 rows=1700 rows_per_s=66.8 devices=8
TIMING stage=prepare_total elapsed_s=38.901 rows=1700 rows_per_s=43.7 limit=50 ...
Dataset saved: rows=1700 path=data/dataset
```

**Primary metric**: `prepare_total` elapsed_s — lower is better.

**Secondary metrics** (for diagnosing where time goes):
- `normalize_total` — CPU normalization time.
- `embed_total` — GPU teacher embedding time (includes model load).
- `embed_teacher_load` — teacher model load time.
- `embed_compute` — pure embedding inference time.
- `normalize_generate` — normalization worker time (excludes merge/save).

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

Example:

```
commit	prepare_s	normalize_s	embed_s	rows	status	description
a1b2c3d	38.901	8.123	25.456	1700	keep	baseline
b2c3d4e	34.210	5.891	23.100	1700	keep	batch normalize buffer from 512 to 2048
c3d4e5f	36.500	4.200	27.800	1650	discard	row count changed (1650 != 1700)
d4e5f6g	0.000	0.000	0.000	0	crash	OOM in teacher embedding
```

## The experiment loop

LOOP FOREVER:

1. Look at the current git state and results so far.
2. Edit one or more of the allowed files with an optimization idea.
3. `git commit -m "description of change"`
4. Run: `uv run prepare --limit 25 --no-stream --rebuild-normalized --detailed-timings > run.log 2>&1`
5. Extract results: `grep "TIMING stage=\(normalize_total\|embed_total\|prepare_total\)" run.log` and `grep "Dataset saved: rows=" run.log`
6. If grep output is empty, the run crashed. `tail -n 50 run.log` to diagnose.
7. Record results in `prep_results.tsv`.
8. **If `prepare_total` improved AND row count matches baseline**: keep the commit, advance the branch.
9. **If `prepare_total` is equal or worse, OR row count changed**: `git reset --hard HEAD~1` to discard.

## Optimization ideas to consider

- Normalization buffer sizes and batch sizes.
- Arrow writer batch sizes.
- Image processing (resize strategy, decode timing).
- Tokenizer call patterns (batching, caching).
- Teacher embedding batch size vs memory tradeoff.
- Teacher model dtype or compilation (`torch.compile`).
- Reducing unnecessary data copies or conversions.
- More efficient Arrow serialization.
- Process pool vs thread pool tradeoffs.
- Reducing per-worker startup overhead.

## Important rules

- **NEVER STOP**. Once the loop begins, do not pause to ask the human. They may be away. Run indefinitely until manually interrupted.
- **Do not modify frozen files**. Ever.
- **Do not change the benchmark command**. The command is the ground truth.
- **Row count is sacred**. Same rows or discard.
- **Simplicity matters**. A tiny speedup that adds ugly complexity is not worth it. Removing code for equal speed is a win.
