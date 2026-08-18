# AetherSearch

AetherSearch is a search-augmented post-training project organized around SFT,
DPO, and reinforcement learning stages.

## Repository Layout

- `sft_data/`: SFT data release metadata and build scripts. The full SFT JSONL
  payload is hosted on Hugging Face rather than committed to GitHub.
- `src/agentic_rl/`: RL training, rollout, advantage, policy loss, retriever,
  checkpoint, and runtime adapter code.
- `scripts/`: launch, preflight, resume, validation, and operational scripts for
  the RL training stage.
- `configs/`: base, formal, hardware, retriever, and executed resolved configs.
- `runtime_assets/`: local runtime assets required by the training launcher.
- `tests/`: unit and integration checks for the training code.
- `environment/`: observed package versions and environment template.

## Model

The model weights are released separately on Hugging Face:
https://huggingface.co/muradil211/AetherSearch

## Training

The production RL entrypoint is:

```bash
bash scripts/_run_runtime_job.sh FORMAL <resolved_config.yaml> <run_dir> [resume_checkpoint]
```

For the preserved formal configs and launch commands, see
`configs/formal_resolved/` and `TRAINING_REPRODUCTION.md`.

Large model weights, optimizer-state checkpoints, eval result bundles, report
archives, and runtime snapshots are intentionally not committed to this GitHub
repository.
