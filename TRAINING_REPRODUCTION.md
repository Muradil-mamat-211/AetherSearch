# Training Reproduction Notes

The production entrypoint used by the persisted formal runs was:

```bash
bash scripts/_run_runtime_job.sh FORMAL <resolved_config.yaml> <run_dir> [resume_checkpoint]
```

`_run_runtime_job.sh` reads the Retriever and RL GPU assignments from the
resolved config, starts both services, then runs:

```bash
python -m agentic_rl.runtime.entrypoint --config <resolved_config.yaml>
```

The package exposes the same runtime adapter as `aethersearch-runtime`, but the
formal training runs used the shell supervisor above so the Retriever and
training-time evaluation worker shared its lifecycle.

`scripts/launch_train.sh` is a compatibility wrapper around the public
`scripts/train_rl.sh` entrypoint. For exact reproduction of a persisted run,
use `_run_runtime_job.sh` with its resolved config.

The actual resolved configs and launch commands preserved for the primary
formal run chain are under `configs/formal_resolved/<run>/`.

Primary formal run chain:

1. `formal_u000_answer_ragen2_paper_mica_ig_v1_g16_20260811_130634`
2. `formal_resume_u040_to_u500_answer_ragen2_mica_ig_v1_g16_20260812_030537`
3. `formal_resume_u180_to_u500_answer_ragen2_mica_ig_v1_g16_20260813_004642`
4. `formal_resume_u320_to_u500_answer_ragen2_mica_ig_v1_g16_20260814_183801`

Eval result bundles, report archives, and training run snapshots were removed
from this code-only GitHub package.
