# Scripts

## Public Entrypoints

- `train_rl.sh`: resolve and launch the single supported RL recipe.
- `resume_rl.sh`: resume a checkpoint only after distributed restore
  validation.
- `validate_static.sh`: lightweight source, shell, and configuration checks.
- `test_code.sh`: run selected tests, or all test modules in isolated
  processes when no paths are supplied.
- `preflight_mica_formal.py`: validate the resolved RL contract before launch.
- `validate_48cpu_resource_profile.py`: check the supported three-rank
  migration profile without starting a training job.

Training-time evaluation is launched automatically by the runtime supervisor
on physical GPU 0. `async_eval_gpu0_worker.sh` is an internal worker entrypoint;
this release does not yet expose a separate one-command standalone model-test
launcher.

`launch_retriever.sh`, `_run_runtime_job.sh`, and
`async_eval_gpu0_worker.sh` are internal components called by the public
launchers. They read paths and GPU assignments from the resolved configuration.

`runtime_guard.py` is the retained monitor/watchdog process used by verified
resume runs. The older standalone monitor/watchdog modules are not part of the
public runtime path.

One-off audit, historical-server, pilot, and smoke-stage scripts are not part
of the public release surface. Their behavior is covered by the source tests
and the archived pre-release branch rather than by extra launch commands.
