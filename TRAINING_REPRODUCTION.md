# Training Reproduction Notes

The production entrypoint used by the persisted formal runs was:

```bash
bash scripts/_run_runtime_job.sh FORMAL <resolved_config.yaml> <run_dir> [resume_checkpoint]
```

`_run_runtime_job.sh` starts the GPU0 retriever, starts the async eval worker, binds RL training to `CUDA_VISIBLE_DEVICES=1,2,3`, then runs:

```bash
python -m agentic_rl.runtime.entrypoint --config <resolved_config.yaml>
```

The package entrypoint is also exposed by `pyproject.toml` as `agentic-rl-train = agentic_rl.controller.update_controller:main`, but the actual formal training runs used the runtime adapter path above.

`scripts/launch_train.sh` is a thin wrapper around
`scripts/train_formal_manual.sh`. For exact reproduction of a persisted run, use
`_run_runtime_job.sh` with a resolved config.

The actual resolved configs and launch commands preserved for the primary
formal run chain are under `configs/formal_resolved/<run>/`.

Primary formal run chain:

1. `formal_u000_answer_ragen2_paper_mica_ig_v1_g16_20260811_130634`
2. `formal_resume_u040_to_u500_answer_ragen2_mica_ig_v1_g16_20260812_030537`
3. `formal_resume_u180_to_u500_answer_ragen2_mica_ig_v1_g16_20260813_004642`
4. `formal_resume_u320_to_u500_answer_ragen2_mica_ig_v1_g16_20260814_183801`

Eval result bundles, report archives, and training run snapshots were removed
from this code-only GitHub package.
