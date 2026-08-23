# Configuration System

The supported public entry configuration is
`recipes/rl/train_4x48gb.yaml`. It composes the training experiment, external
assets, hardware resources, runtime mapping, and official qualification
contract. Machine-local paths come from `environment/env.local.sh`.

The public prompt filter ranks candidate prompts by raw terminal-outcome sample
variance and retains the shortest prefix carrying the configured cumulative
variance mass. Retrieval scoring and Search credit are applied only after that
filtering step.

## Configuration Layers

| Layer | Responsibility |
|---|---|
| experiment | algorithm and training schedule |
| assets | model, dataset, tokenizer, corpus, and index identities |
| hardware | physical GPUs, nodes, CPU/RAM, and role placement |
| runtime | Ray and backend resource mapping |
| qualification | exact contract for the verified reference run |

`base.yaml` and inherited experiment YAML files are abstract layers. They do
not select a GPU count, CUDA mapping, Ray capacity, rollout replica layout,
GPU-memory limit, or learner micro-batch capacity. Those values come from an
explicit hardware/runtime profile.

## Topology Resolution

The hardware profile supplies one topology input. `TopologyPlan` then derives:

- visible CUDA device mappings;
- learner world size;
- Ray placement bundles;
- rollout data/tensor parallel compatibility fields;
- veRL `nnodes` and `n_gpus_per_node`.

Derived compatibility fields are materialized into the resolved configuration;
they are not independent topology sources. Backend-specific validation reports
unsupported parallel layouts separately from algorithm validation.

## Qualified and Portable Modes

The repository has one verified reference topology: four 48 GiB GPUs, with one
GPU assigned to retrieval/asynchronous evaluation and three GPUs assigned to
the RL runtime. `configs/qualification/` contains the opt-in exact contract for
that environment.

Portable mode validates generic resource minimums and topology invariants.
CPU-only synthetic layouts test configuration derivation, not GPU-memory fit,
runtime stability, throughput, convergence, or production qualification.

## Assets and Audit Inputs

`configs/assets/` records external paths and SHA-256 identities independently
from the training experiment. Replacing a model or dataset therefore requires
a new asset manifest rather than a copy of the algorithm configuration.

Retrieval-scoring audit artifacts are selected through
`AETHERSEARCH_EXACT_IG_AUDIT_ROOT`. They are executable preflight inputs, not
embedded server paths or design documents.

Other root YAML files preserve inherited experiment, retriever, schedule,
capacity, and historical compatibility layers. Public launches should resolve
the recipe rather than invoke those layers directly.
