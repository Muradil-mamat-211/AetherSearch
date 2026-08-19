# RL Recipe

This directory contains the public entry configuration for the verified
AetherSearch RL topology. It does not contain SFT or DPO training recipes.

The current reference profile uses four 48 GiB GPUs: physical GPU 0 hosts the
hybrid retriever and asynchronous worker, while physical GPUs 1-3 host the
three-rank vLLM/FSDP2 RL runtime. Other topologies are not yet claimed as
validated.

Prepare a local environment file without committing machine paths:

```bash
cp environment/env.template.sh environment/env.local.sh
# Edit environment/env.local.sh, then:
source environment/env.local.sh
```

Resolve and validate the recipe without starting services:

```bash
bash scripts/train_rl.sh --dry-run
```

Start the verified RL run:

```bash
bash scripts/train_rl.sh
```

Resume only from a checkpoint that has passed the repository's fresh-runtime
distributed restore validation:

```bash
bash scripts/resume_rl.sh \
  --config recipes/rl/train_4x48gb.yaml \
  --checkpoint /path/to/checkpoint \
  --restore-validation /path/to/restore_validation.json \
  --source-run /path/to/source_run
```

By default, the launcher creates a timestamped run below
`AETHERSEARCH_RUNTIME_ROOT`. Use `--run-dir PATH` to select an exact run
directory. The launcher materializes `configs/resolved_config.yaml` there,
runs the formal preflight, then starts the Retriever, asynchronous
training-time evaluation worker, and three-rank RL runtime from the same
resolved config.

Users with a different topology should copy this recipe and hardware profile,
then update the hardware/Ray fields together. Only the included 4x48GB profile
is asserted as validated by this release.
