# optimize.md

Find the optimal `data.batch_size` in `config.toml` for the GPU(s) on this machine.

## Goal

Maximize teacher embedding throughput (rows_per_s) by finding the highest `data.batch_size` that doesn't OOM. Images are already capped at `max_image_tokens = 1280` during normalization, so per-sample cost is bounded — this is purely a batch size search.

**Metric**: `embed_compute` rows_per_s from detailed timings. Higher is better.

## Setup

1. **Warm the caches first**: Run `uv run prepare --limit 50` once to cache raw data and normalized samples.
2. **Check GPU info**: Run `nvidia-smi` to identify the card(s) and VRAM.
3. **Initialize batch_results.tsv** with the header row.
4. **Confirm and go**.

## The benchmark command

Always run exactly this (substituting BATCH for the batch size being tested):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run prepare --limit 50 --detailed-timings > run.log 2>&1
```

Before each run, update `data.batch_size` in `config.toml` to the value being tested.

This ensures:
- Raw data and normalized cache are reused (only embedding re-runs).
- Detailed timing lines are emitted for comparison.
- `expandable_segments` avoids fragmentation OOMs.

## What you CAN modify

- `config.toml` — only the `data.batch_size` value.

## What you CANNOT modify

- Any source code files.
- Any other config values.

## Reading results

After a run, extract:

```bash
grep "TIMING stage=embed_compute\|TIMING stage=embed_total\|TIMING stage=prepare_total" run.log
grep "GPU utilization peaked" run.log
grep "Dataset saved: rows=" run.log
```

**Primary metric**: `embed_compute` rows_per_s — higher is better.
**Secondary**: GPU utilization % — should be as high as possible without OOM.

## Logging results

Log to `batch_results.tsv` (tab-separated).

Header and columns:

```
batch_size	embed_s	rows_per_s	gpu_pct	rows	status	notes
```

- `batch_size`: the value tested
- `embed_s`: embed_compute elapsed_s
- `rows_per_s`: embed_compute rows_per_s
- `gpu_pct`: peak GPU utilization %
- `rows`: final row count (must match baseline)
- `status`: `ok`, `oom`, or `crash`
- `notes`: observations

## The search loop

LOOP:

1. Pick the next `data.batch_size` to test based on results so far.
2. Update `config.toml` with the new batch_size.
3. Run the benchmark command.
4. Extract results. If grep output is empty, check `tail -n 50 run.log` for OOM or crash.
5. Record in `batch_results.tsv`.
6. Repeat.

### Search strategy

Start with a binary search:
1. Begin at `batch_size = 64` (known safe baseline).
2. Double it: 128, 256, 512... until OOM.
3. Once OOM is hit, binary search between last-ok and first-oom.
4. Narrow to within ±8 of the sweet spot.
5. Run the winner 2-3 times to confirm it's stable (not flaky OOM).

### When to stop

Stop when you've found a stable batch_size that:
- Doesn't OOM across 3 consecutive runs.
- Is within 8 of the OOM boundary (diminishing returns beyond this).
- Report the recommended value clearly at the end.

## Correctness rule

Row count must match the baseline exactly. If it changes, the run is invalid.

## Important rules

- **NEVER STOP** until the search is complete. The human may be away.
- **Only modify `data.batch_size`** in config.toml. Nothing else.
- **Row count is sacred**. Same rows or discard.
- **If OOM**: record it, halve the step, try again. Don't panic.
- **Restore config.toml** to the best-found batch_size when done.
