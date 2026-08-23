# Operational Scripts

## Public Entrypoints

- `train_rl.sh`: resolve, preflight, and launch the supported RL recipe.
- `resume_rl.sh`: resume a checkpoint only after distributed restore
  validation.
- `validate_static.sh`: lightweight source, shell, and configuration checks.
- `validate_readme.py`: strict UTF-8, Markdown fence, link, anchor, table, and
  repository-owned SVG checks for the root README.
- `test_code.sh`: run selected tests, or all test modules in isolated
  processes when no paths are supplied.
- `validate_48cpu_resource_profile.py`: check the supported three-rank
  migration profile without starting a training job.

The training launcher owns configuration resolution and formal preflight. Users
do not need to invoke its Python helpers directly.

## Runtime Components

Training-time evaluation is launched automatically by the runtime supervisor
using the configured `eval` role. `async_eval_worker.sh` is an internal worker
entrypoint; this release does not expose a separate one-command standalone
model-test launcher.

`launch_retriever.sh`, `_run_runtime_job.sh`, and `async_eval_worker.sh` are
internal components called by the public
launchers. They read paths and GPU assignments from the resolved configuration.

`runtime_guard.py` is the retained monitor/watchdog process used by verified
resume runs. The older standalone monitor/watchdog modules are not part of the
public runtime path.

## Historical Utilities

One-off audit, historical-server, pilot, and smoke-stage scripts are not part
of the public release surface. Their behavior is covered by the source tests
and the archived pre-release branch rather than by extra launch commands.
