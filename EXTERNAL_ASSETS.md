# External Assets

These paths were referenced by the resolved training configuration or launch scripts on this machine. They are not copied into the GitHub code bundle.

## Base/Reference Model

- Actor/reference start model: `/root/autodl-tmp/search-r1-workspace/models/dpo_v2_final_model`

## Training And Validation Data

- Train data: `/root/autodl-tmp/search-r1-workspace/data/nq_hotpotqa_train/train.parquet`
- Validation data: `/root/autodl-tmp/search-r1-workspace/data/nq_hotpotqa_train/test.parquet`

## Retriever Assets

- BM25 index: `/root/autodl-tmp/search-r1-workspace/data/wiki18_bm25/bm25`
- Dense FAISS index: `/root/autodl-tmp/search-r1-workspace/data/nq_search/e5_Flat.index`
- Corpus: `/root/autodl-tmp/search-r1-workspace/data/wiki18_corpus/wiki-18.jsonl`
- Dense retriever model: `/root/autodl-tmp/search-r1-workspace/models/e5-base-v2`

## External Code/Packages

- Installed `verl==0.6.1` was used from the RL environment site-packages.
- Local Search-R1 tree exists at `/root/autodl-tmp/search-r1-workspace/code/Search-R1`, commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`, with local modifications, but `scripts/bootstrap_env.sh` intentionally does not put that tree on `PYTHONPATH` because it shadows installed veRL.
- Retriever server used by this project is vendored as `runtime_assets/retriever/hybrid_retrieval_server.py`.
