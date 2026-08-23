# Third-Party Audit Material

This directory is limited to attribution, license, and parity-audit material.
It includes a minimal IGPO official source snapshot under
`igpo_official_64165e2741ed8801f977948c8128080ce87b4101/` for audit and parity
reference. The training runtime does not import that snapshot.

No model weights, dataset payloads, retrieval indexes, or additional external
source trees are vendored here. Runtime dependencies are installed packages;
large artifacts are resolved through the project asset manifest and
machine-local environment configuration.

The project retrieval-scoring implementation is independent of the audit
snapshot. Attribution, pinned revision, reviewed source hashes, and license
details are recorded in `../THIRD_PARTY_NOTICES.md`.
