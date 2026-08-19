# External Assets

These large assets are not copied into the GitHub code bundle. Configure their
local locations in `environment/env.local.sh`.

## Base/Reference Model

- Actor/reference start model: `AETHERSEARCH_ACTOR_MODEL` and
  `AETHERSEARCH_REFERENCE_MODEL`

## Training And Validation Data

- Train data: `AETHERSEARCH_TRAIN_DATA`
- Validation data: `AETHERSEARCH_VALIDATION_DATA`; download the complete
  Search-R1 test parquet from
  [muradil211/AetherSearch-Eval](https://huggingface.co/datasets/muradil211/AetherSearch-Eval)

## Retriever Assets

- BM25 index: `AETHERSEARCH_BM25_INDEX_PATH`
- Dense FAISS index: `AETHERSEARCH_DENSE_INDEX_PATH`
- Corpus: `AETHERSEARCH_CORPUS_PATH`
- Dense retriever model: `AETHERSEARCH_DENSE_ENCODER_PATH`

## External Code/Packages

- Installed `verl==0.6.1` was used from the RL environment site-packages.
- The released run used Search-R1 commit
  `598e61bd1d36895726d28a8d06b3a15bed19f5d3`; set its location with
  `AETHERSEARCH_SEARCH_R1_ROOT`. `scripts/bootstrap_env.sh` intentionally does
  not put this tree on `PYTHONPATH` because it can shadow the installed veRL.
- Retriever server used by this project is vendored as `runtime_assets/retriever/hybrid_retrieval_server.py`.
