# Third-party code policy

This release includes a minimal IGPO official source snapshot under
`igpo_official_64165e2741ed8801f977948c8128080ce87b4101/` for audit and parity
reference. The training runtime does not import that snapshot.

No model, dataset index, Retriever index, or external third-party source tree is
vendored here. The project imports installed packages and references existing
assets by absolute, read-only paths.

The Exact-IG task construction is an independent implementation informed by the
official IGPO repository. Attribution, pinned revision, source hashes, and license
are recorded in `../THIRD_PARTY_NOTICES.md`.
