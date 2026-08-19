# RL Training Reproduction

The supported public entrypoint is:

```bash
bash scripts/train_rl.sh
```

The launcher resolves `recipes/rl/train_4x48gb.yaml`, writes an immutable
`configs/resolved_config.yaml` inside the new run directory, performs the
formal preflight, and then starts the Retriever, asynchronous full-data eval
worker, and RL runtime supervisor.

Every 20 successful updates, the runtime exports a model and queues evaluation
over all 51,713 rows of the configured Search-R1 `test.parquet`. The same full
manifest is used at every cadence point through update 500.

The internal runtime command is:

```bash
python -m agentic_rl.runtime.entrypoint --config <resolved_config.yaml>
```

Use `scripts/resume_rl.sh` only with a checkpoint that has passed the required
fresh-runtime distributed restore validation.
