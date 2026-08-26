# Retriever Runtime Asset

This directory contains the retrieval service implementation and its default
service configuration:

| File | Purpose |
|---|---|
| `hybrid_retrieval_server.py` | hybrid sparse/dense retrieval server |
| `retriever.yaml` | immutable release service asset (checksum input) |

Model, corpus, and index artifacts are external inputs configured through
`environment/env.local.sh` and the asset manifest. GPU placement is derived
from the resolved topology rather than fixed in this directory.
Pinned upstream revisions, download and assembly commands, expected sizes, and
checksums are documented in `../../EXTERNAL_ASSETS.md`. Those assets remain in
their upstream repositories and are not copied into this GitHub repository.

Use `scripts/launch_retriever.sh` through the public training launcher; the
server file is a runtime component, not a standalone reproduction entrypoint.
Active backend policy is owned by `runtime.retriever` in the composed runtime
profile; the immutable YAML remains only as a release/checksum asset and is not
read as a second launcher policy source.
