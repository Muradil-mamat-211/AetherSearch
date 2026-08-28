# Runtime Versions Observed Locally

RL/training environment used for the released run:

```text
python 3.12.13
torch 2.8.0+cu128
FSDP2 torch.distributed (bundled with PyTorch)
ray 2.51.2
transformers 4.57.1
vllm 0.11.0
verl 0.6.1
numpy 1.26.4
pandas 3.0.5
pyarrow 25.0.0
aiohttp 3.14.3
PyYAML 6.0.3
safetensors 0.8.0
```

Retriever environment used for the released run:

```text
python 3.10.20
torch 2.11.0+cu128
transformers 4.47.1
pyserini 1.2.0
faiss 1.8.0
Flask 3.1.3
FastAPI 0.136.3
uvicorn 0.48.0
numpy 1.26.4
```

SFT-2000 code-audit and complete data/mask preflight environment observed on
2026-08-28 UTC (no CUDA training run was performed in this environment):

```text
python 3.10.20
torch 2.12.1+cu130
transformers 4.51.3
accelerate 1.14.0
deepspeed 0.19.2
tensorboard 2.20.0
```
