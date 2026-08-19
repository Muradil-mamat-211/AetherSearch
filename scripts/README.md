# Scripts

## Public Entrypoints

- `train_rl.sh`: resolve and launch the single supported RL recipe.
- `resume_rl.sh`: resume a checkpoint only after distributed restore
  validation.
- `validate_static.sh`: lightweight source, shell, and configuration checks.
- `test_code.sh`: run selected tests, or all test modules in isolated
  processes when no paths are supplied.

Training-time evaluation is launched automatically by the runtime supervisor
on physical GPU 0. `async_eval_gpu0_worker.sh` is an internal worker entrypoint;
this release does not yet expose a separate one-command standalone model-test
launcher.

`launch_retriever.sh` and the underscore-prefixed process scripts are internal
building blocks used by the public launchers. They read machine paths and GPU
assignments from the resolved configuration.

## Historical Operations

The remaining launch, gate, audit, monitoring, and recovery scripts preserve
the exact operational code used while qualifying the released run. Some of
these one-off scripts intentionally retain original server paths or fixed
hardware assertions and are not alternative public recipes.

Offline eval/report generation scripts and generated eval/report artifacts are
outside this code release. Use `recipes/rl/train_4x48gb.yaml` as the supported
starting point rather than a historical one-off launcher.
