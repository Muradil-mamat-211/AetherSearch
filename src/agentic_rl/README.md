# Agentic RL Package

This package implements the public AetherSearch RL stage. Its module boundaries
separate trajectory generation, scoring and credit assignment, optimization,
checkpointing, and infrastructure integration.

| Area | Responsibility |
|---|---|
| `rollout/` | trajectory schema, Search/Answer turns, and token provenance |
| `outcome/` | terminal parsing, format checks, and alias-aware token F1 |
| `selection/` | prompt scoring and cumulative mass filtering |
| `exact_ig/` | canonical-answer retrieval utility scoring |
| `advantage/` | Search and Answer credit assignment |
| `policy/` | turn ratios, clipping, KL, and balanced reduction |
| `controller/` | prompt sampling and transactional update control |
| `checkpoint/` | atomic and distributed checkpoint contracts |
| `retriever/` | retrieval protocol, client, and health checks |
| `runtime/` | Ray, veRL, vLLM, FSDP2, evaluation, and launcher adapters |
| `workers/` | distributed worker and resource integration |

`topology.py` owns the derived runtime topology. Algorithm configuration,
assets, hardware resources, runtime mapping, and official qualification remain
separate inputs.

Historical internal identifiers are retained where required for configuration
and checkpoint compatibility. Public method definitions are documented in the
root README.
