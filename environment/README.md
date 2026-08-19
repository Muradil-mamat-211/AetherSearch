# Runtime Environments

The released topology uses two Python environments, not one environment per
engine:

1. The RL environment runs the controller, Ray workers, vLLM rollout engine,
   FSDP2 Actor/Reference scoring, and FSDP2 training engine. Rollout and
   training are colocated on the same three RL GPUs and must use one compatible
   PyTorch/veRL/vLLM package set.
2. The Retriever environment runs Pyserini, FAISS GPU, the dense encoder, and
   the hybrid retrieval server on the dedicated Retriever GPU.

FSDP2 is not a separate pip distribution. It is provided by
`torch.distributed` in `torch==2.8.0+cu128`; veRL selects the `fsdp2` strategy,
and `src/agentic_rl/runtime/fsdp_worker.py` supplies the project-specific
training worker. Therefore the train engine dependency appears as `torch` and
`verl` in the environment inventory, not as a package named `fsdp`.

`RUNTIME_VERSIONS.md` records the principal versions, and the two pip freeze
files preserve the observed package inventories. They are provenance
snapshots, not fully portable lock files: the RL freeze includes a local
FlashAttention wheel and the original editable project path.

Strict environment reproduction therefore still requires a published
container image or portable lock files plus CUDA/NCCL/JDK build details. The
current public launcher selects the two interpreters with
`AETHERSEARCH_RL_PYTHON` and `AETHERSEARCH_RETRIEVER_PYTHON`.
