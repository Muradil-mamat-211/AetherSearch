# Retriever Runtime Asset

This directory contains the retrieval service implementation and its default
service configuration:

| File | Purpose |
|---|---|
| `hybrid_retrieval_server.py` | hybrid sparse/dense retrieval server |
| `retriever.yaml` | service defaults consumed by the launcher |

Model, corpus, and index artifacts are external inputs configured through
`environment/env.local.sh` and the asset manifest. GPU placement is derived
from the resolved topology rather than fixed in this directory.

Use `scripts/launch_retriever.sh` through the public training launcher; the
server file is a runtime component, not a standalone reproduction entrypoint.
