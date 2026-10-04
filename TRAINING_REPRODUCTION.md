# Training Reproduction

## SFT-2600

The strict SFT implementation is documented in [`sft/`](sft/). Download the
frozen 2,600-record dataset from
[muradil211/AetherSearch_SFT](https://huggingface.co/datasets/muradil211/AetherSearch_SFT),
install `sft/requirements.txt`, run the data-only preflight, and then launch the
single-node BF16 ZeRO-3 recipe:

```bash
bash sft/scripts/run_train_sft_2600_zero3.sh
```

This is one public SFT stage: the pinned Qwen base model is supervised on the
frozen 2,600-record full-trajectory dataset and exported to `final_model/`.
The [AetherSearch SFT repository](https://huggingface.co/muradil211/AetherSearch_SFT)
hosts the released SFT model artifacts.

The launcher discovers the number of visible GPUs and derives gradient
accumulation to preserve global batch 24. It does not assign GPU IDs or embed
machine-local paths. Topology and paths are supplied through environment
variables; the 2,600-record count and dataset SHA-256 remain fixed.

## DPO

For new or regenerated decision-level preference data, follow the
[DPO construction specification](dpo/README.md#dpo-data-construction-workflow):
complete rollouts establish terminal correctness, and a separate action check
establishes the local preference. The training commands below reproduce the
existing frozen release using its recorded count and checksum.

The strict preference-training implementation is documented in
[`dpo/`](dpo/). Download the canonical 2,126-pair `train.jsonl` from
[muradil211/AetherSearch_DPO](https://huggingface.co/datasets/muradil211/AetherSearch_DPO),
install `dpo/requirements.txt`, run the data-only preflight, and start the
single-node BF16 ZeRO-3 recipe:

```bash
bash dpo/scripts/run_train_dpo_zero3.sh
```

This is one public DPO stage: the pinned AetherSearch SFT checkpoint is used
for both the initial policy and frozen reference, all canonical preference
pairs are consumed without truncation, and the result is exported atomically
to `final_model/`. The resulting checkpoint is released at
[muradil211/AetherSearch_DPO](https://huggingface.co/muradil211/AetherSearch_DPO).

The launcher discovers already-visible GPUs and derives gradient accumulation
to preserve global batch 12. It does not assign GPU IDs, embed machine-local
paths, or set host-specific communication and allocator policy. The pair count
and dataset SHA-256 remain fixed.

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
over all 1,400 rows of `eval_1400.jsonl` from
[muradil211/AetherSearch_Eval_1400](https://huggingface.co/datasets/muradil211/AetherSearch_Eval_1400).
The same frozen manifest is used at every cadence point through update 500.

The internal runtime command is:

```bash
python -m agentic_rl.runtime.entrypoint --config <resolved_config.yaml>
```

Use `scripts/resume_rl.sh` only with a checkpoint that has passed the required
fresh-runtime distributed restore validation.
