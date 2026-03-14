# prep-data.md

Run a full data preparation job and babysit it to completion. The human is away — you are responsible for making sure this succeeds.

## The command

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run prepare --limit 10000 --push-to-hub --detailed-timings > prep.log 2>&1
```

This will take a long time. It downloads raw data, normalizes it, generates teacher embeddings on GPU, saves the dataset, and pushes to Hugging Face Hub.

## Your job

1. **Start the run.**
2. **Monitor it periodically.** Check `tail -n 30 prep.log` every few minutes. Look for:
   - Progress: `Normalized ...` lines (normalization phase), `Embedded ...` or `TIMING stage=embed_shard` lines (embedding phase).
   - Errors: tracebacks, OOM, network failures, disk full, broken pipe.
   - Stalls: if the last log line hasn't changed in 10+ minutes, something may be stuck.
3. **If it crashes, fix it and restart.** Common issues:
   - **CUDA OOM**: Lower `data.batch_size` in `config.toml` (try halving it) and rerun.
   - **Network error during download**: Rerun — raw cache is incremental, it picks up where it left off.
   - **Network error during push-to-hub**: The dataset is saved locally to `data/dataset/` first. Manually push with `uv run python -c "from datasets import Dataset; Dataset.load_from_disk('data/dataset').push_to_hub('ljt019/delta-embed-vl-10000', split='train')"`.
   - **Disk full**: Check `df -h`. Clear `data/raw/` caches for already-normalized sources if needed.
   - **Normalization failure on a specific source**: Check the traceback. If one source is broken, note it and move on — the rest of the data is still valuable. If it's a transient error, retry.
   - **BrokenProcessPool / unpicklable exception**: Already handled in the code, but if it resurfaces, check the inner error message.
4. **If it succeeds**, verify:
   - `grep "Dataset saved: rows=" prep.log` — confirm a reasonable row count (should be tens of thousands).
   - `grep "Pushing dataset to Hub" prep.log` — confirm hub push happened.
   - `grep "TIMING stage=prepare_total" prep.log` — note total elapsed time.
5. **Report the result.** Write a short summary to `prep_report.txt` with: row count, total time, any issues encountered, and whether the hub push succeeded.

## Rules

- **Do NOT modify source code** unless something is genuinely broken and needs a fix to proceed.
- **Do NOT change config.toml** unless you need to lower batch_size to avoid OOM.
- **Do NOT stop.** If something fails, diagnose, fix, and retry. The human is asleep. 
- **Do NOT give up.** If you hit a wall, try a different approach. Lower batch size. Clear caches. Retry. The goal is data on Hub by morning.
- **Be patient.** This run may take hours. That's expected. Don't kill it because it's slow.
