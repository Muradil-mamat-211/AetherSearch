# Training Reproduction

## SFT-2000

The strict SFT implementation is documented in [`sft/`](sft/). Download the
frozen 2,000-record dataset from
[muradil211/AetherSearch_SFT](https://huggingface.co/datasets/muradil211/AetherSearch_SFT),
install `sft/requirements.txt`, run the data-only preflight, and then launch the
single-node BF16 ZeRO-3 recipe:

```bash
bash sft/scripts/run_train_sft_2000_zero3.sh
```

This is one public SFT stage: the pinned Qwen base model is supervised on the
frozen 2,000-record full-trajectory dataset and exported to `final_model/`.
The [AetherSearch-SFT repository](https://huggingface.co/muradil211/AetherSearch-SFT)
is reserved for the checkpoint produced by this recipe.

## Agentic RL

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
