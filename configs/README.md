# Configurations

The supported public entry configuration is
`recipes/rl/train_4x48gb.yaml`. It composes the verified algorithm settings in
this directory with `configs/hardware/4x48gb_3rl.yaml`, while all machine-local
paths come from `environment/env.local.sh`.

`configs/base.yaml` and the inherited algorithm YAMLs are abstract layers; they
intentionally do not select a GPU count, CUDA mapping, Ray capacity, rollout
replica layout, GPU memory limit, or learner micro-batch size. Those values are
provided by an explicit hardware/runtime profile and resolved through
`TopologyPlan`.

Configuration responsibilities are intentionally separate:

- `configs/experiment`-equivalent inherited sections describe the training
  algorithm and schedule.
- `configs/assets/` declares external model/data/index paths and SHA-256 values.
- `configs/hardware/` declares physical GPUs, nodes, CPU/RAM, Ray resource
  capacity, and backend capacity tuning. Its `topology` block is the single
  topology input; derived world size, visible CUDA IDs, and DP/TP compatibility
  fields are materialized by the planner.
- `configs/qualification/` contains the opt-in exact official reproduction
  contract.

The runtime derives visible CUDA IDs, FSDP2 world size, Ray bundles, and veRL
`nnodes`/`n_gpus_per_node` from `TopologyPlan`. The generic validator does not
require the official GPU IDs, world size, CPU count, or node count. The current
veRL/vLLM adapter still reports unsupported TP/DP combinations explicitly;
that is a backend limitation, not an algorithm invariant.

The repository has one validated reference topology: the official 4x48GB
profile. Additional non-reference layouts are covered only by CPU-only
synthetic configuration tests for role mapping and derived topology fields;
those tests do not establish GPU-memory fit, runtime compatibility, training
stability, throughput, or production qualification.

The other root YAML files are inherited algorithm, retriever, schedule, gate,
and historical qualification layers. Public launches should use the recipe,
not invoke those layers independently.

The Exact-IG audit bundle is selected with
`AETHERSEARCH_EXACT_IG_AUDIT_ROOT`; it is an executable preflight input, not a
design document. The public recipe does not embed model-dependent audit
artifacts or server paths.
