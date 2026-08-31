# Runtime Environments

This directory records the observed software environments and provides the
template for machine-local paths. It contains no credentials or committed
`env.local.sh` file.

## Engine Split

The verified reference topology uses two Python environments, not one
environment per engine:

1. The RL environment runs the controller, Ray workers, vLLM rollout engine,
   FSDP2 Actor/Reference scoring, and FSDP2 training engine. Rollout and
   training are colocated on the same three RL GPUs and must use one compatible
   PyTorch/veRL/vLLM package set.
2. The Retriever environment runs Pyserini, FAISS GPU, the dense encoder, and
   the hybrid retrieval server on the dedicated Retriever GPU.

The SFT and DPO trainers are separate, self-contained entrypoints and do not
use the Ray/veRL runtime topology. Install their compact dependency sets from
[`sft/requirements.txt`](../sft/requirements.txt) and
[`dpo/requirements.txt`](../dpo/requirements.txt). Both launchers select an
interpreter with `PYTHON_BIN` and receive model/data/output paths through
explicit environment variables. They discover visible GPUs, derive
accumulation from a fixed effective global batch, and leave GPU selection,
NCCL fabric policy, and allocator tuning to the surrounding runtime. Their
public launchers use a single-node `torchrun` boundary and do not assume a
cluster scheduler or shared filesystem.

FSDP2 is not a separate pip distribution. It is provided by
`torch.distributed` in `torch==2.8.0+cu128`; veRL selects the `fsdp2` strategy,
and `src/agentic_rl/runtime/fsdp_worker.py` supplies the project-specific
training worker. Therefore the train engine dependency appears as `torch` and
`verl` in the environment inventory, not as a package named `fsdp`.

## Local Configuration

Create the ignored local file from the public template:

```bash
cp environment/env.template.sh environment/env.local.sh
```

Set local model, dataset, retriever, runtime-root, and Python interpreter paths
there. Hardware roles and backend capacity remain in YAML; secrets should be
provided through the surrounding process environment and never committed.

## Version Evidence

`RUNTIME_VERSIONS.md` records the principal versions, and the two pip freeze
files preserve the observed package inventories. They are provenance
snapshots, not fully portable lock files: the RL freeze includes a local
FlashAttention wheel and the original editable project path.

Strict environment reproduction therefore still requires a compatible
container or portable lock files plus CUDA/NCCL/JDK build details. The public
launcher selects the two interpreters with
`AETHERSEARCH_RL_PYTHON` and `AETHERSEARCH_RETRIEVER_PYTHON`. The same local
file also selects `AETHERSEARCH_ASSET_MANIFEST` and
`AETHERSEARCH_QUALIFICATION_MODE`; use `reference` for the official profile
and `portable` for a user-defined hardware profile.
