# External Assets

These large assets are not copied into the GitHub code bundle. Configure their
local locations in `environment/env.local.sh`.

## Model Artifact Repositories

- Final AetherSearch model:
  [muradil211/AetherSearch](https://huggingface.co/muradil211/AetherSearch).
- SFT-2000 model output repository:
  [muradil211/AetherSearch-SFT](https://huggingface.co/muradil211/AetherSearch-SFT).
  This repository is designated for the checkpoint produced by the public
  one-stage SFT-2000 trainer.

## Base/Reference Model

- Actor/reference start model: `AETHERSEARCH_ACTOR_MODEL` and
  `AETHERSEARCH_REFERENCE_MODEL`

## Training And Validation Data

- SFT-2000 data:
  [muradil211/AetherSearch_SFT](https://huggingface.co/datasets/muradil211/AetherSearch_SFT),
  file `final_sft_2000.jsonl`, 2,000 records, SHA-256
  `fec609652d3832c7a6c0ee2861c6f946b6cf7c3d3d40fc5d9be9b75df6325dcb`.
- Train data: `AETHERSEARCH_TRAIN_DATA`; use the upstream Search-R1 dataset
  [`PeterJinGo/nq_hotpotqa_train`](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train),
  file `train.parquet`. The released source identity is 169,615 rows and
  SHA-256 `c3cc21e862a8469105de666101578cbff23cdc77e91a803cef102622c89cc4f6`.
- Validation data: `AETHERSEARCH_VALIDATION_DATA`; download the complete
  Search-R1 test parquet from
  [muradil211/AetherSearch-Eval](https://huggingface.co/datasets/muradil211/AetherSearch-Eval)

## Retriever Assets

AetherSearch does not redistribute the approximately 80 GB Retriever asset
set. The official run used the following pinned upstream revisions and local
asset identities. Downloading from `main` without `--revision` is not an exact
reproduction.

### Pinned Reference Asset Set

**Wikipedia corpus**

- Source: [`PeterJinGo/wiki-18-corpus`](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/tree/69c1c00ffe7c5554c68d8548355cb22e46aabc51),
  revision `69c1c00ffe7c5554c68d8548355cb22e46aabc51`.
- Downloaded file: `wiki-18.jsonl.gz`, 5,123,307,260 bytes, SHA-256
  `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db`.
- Runtime file after decompression: `wiki-18.jsonl`, 14,393,579,520 bytes,
  21,015,324 passages, SHA-256
  `85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9`.
- Environment variable: `AETHERSEARCH_CORPUS_PATH`.

**BM25 Lucene index**

- Source: [`PeterJinGo/wiki-18-bm25-index`](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index/tree/2c7554f25f425038c4bcb155735a0f831851fd78),
  revision `2c7554f25f425038c4bcb155735a0f831851fd78`.
- Runtime directory: `bm25/`, 2,297,615,435 bytes.
- Canonical project tree SHA-256:
  `d5635cabf617e999749697bb0fa686ab4509ba9c4bd33be7d52c6187cc956724`.
- Environment variable: `AETHERSEARCH_BM25_INDEX_PATH`.

**Dense E5 FAISS index**

- Source: [`PeterJinGo/wiki-18-e5-index`](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/tree/a4d31160a035f30764604f4827cd8f1d0315eb86),
  revision `a4d31160a035f30764604f4827cd8f1d0315eb86`.
- `part_aa`: 42,949,672,960 bytes, SHA-256
  `a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023`.
- `part_ab`: 21,609,402,413 bytes, SHA-256
  `b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11`.
- Runtime file after ordered concatenation: `e5_Flat.index`, 64,559,075,373
  bytes, SHA-256
  `69c98463fdb41fc08737d88513c597725f311c44f7ba4dca4b05d8c7c658d166`.
- Expected structure: `IndexFlatIP`, 21,015,324 vectors, dimension 768. The
  qualified runtime streams this flat index to a GPU as `GpuIndexFlatIP`.
- Environment variable: `AETHERSEARCH_DENSE_INDEX_PATH`.

**Dense query encoder**

- Source: [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2/tree/f52bf8ec8c7124536f0efb74aca902b2995e5bcd),
  revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`.
- Runtime weights: `model.safetensors`, 437,955,512 bytes, SHA-256
  `d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693`.
- Runtime `config.json` SHA-256:
  `01cc39aa39538a8179aa131c8a16adc03b506c92009f091502e4eb0c702f5f78`.
- Project tokenizer SHA-256 (computed by `tokenizer_sha256()`):
  `f529fd7999007e7ef1bf956fcd222da1e613d5a59e10392e213d905a085c6adf`.
- Environment variable: `AETHERSEARCH_DENSE_ENCODER_PATH`.

The BM25 value above is not a tarball hash. It is produced by
`agentic_rl.assets.sha256_tree()`, which hashes each file in sorted relative
path order together with its relative path and byte size. The tokenizer value
is produced by `agentic_rl.assets.tokenizer_sha256()` over the runtime
tokenizer files.

### Download And Assemble

The following commands download from the upstream repositories; they do not
upload or copy these assets into AetherSearch:

```bash
retriever_root=/path/to/aethersearch-retriever
mkdir -p "${retriever_root}/corpus" \
  "${retriever_root}/bm25-download" \
  "${retriever_root}/dense" \
  "${retriever_root}/e5-base-v2"

hf download PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz \
  --repo-type dataset \
  --revision 69c1c00ffe7c5554c68d8548355cb22e46aabc51 \
  --local-dir "${retriever_root}/corpus"
gzip -dk "${retriever_root}/corpus/wiki-18.jsonl.gz"

hf download PeterJinGo/wiki-18-bm25-index \
  --repo-type dataset \
  --revision 2c7554f25f425038c4bcb155735a0f831851fd78 \
  --local-dir "${retriever_root}/bm25-download"

hf download PeterJinGo/wiki-18-e5-index part_aa part_ab \
  --repo-type dataset \
  --revision a4d31160a035f30764604f4827cd8f1d0315eb86 \
  --local-dir "${retriever_root}/dense"
cat "${retriever_root}/dense/part_aa" \
  "${retriever_root}/dense/part_ab" \
  > "${retriever_root}/dense/e5_Flat.index"

hf download intfloat/e5-base-v2 \
  config.json model.safetensors tokenizer.json tokenizer_config.json \
  special_tokens_map.json vocab.txt \
  --revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
  --local-dir "${retriever_root}/e5-base-v2"
```

Point the local environment at the resulting paths:

```bash
export AETHERSEARCH_CORPUS_PATH=${retriever_root}/corpus/wiki-18.jsonl
export AETHERSEARCH_BM25_INDEX_PATH=${retriever_root}/bm25-download/bm25
export AETHERSEARCH_DENSE_INDEX_PATH=${retriever_root}/dense/e5_Flat.index
export AETHERSEARCH_DENSE_ENCODER_PATH=${retriever_root}/e5-base-v2
```

The official asset manifest at
`configs/assets/aethersearch_release_v1.yaml` verifies these local runtime
files. In addition, Retriever startup fails closed unless the corpus passage
count equals the FAISS vector count and the encoder output dimension equals the
FAISS dimension.

## External Code/Packages

- Installed `verl==0.6.1` was used from the RL environment site-packages.
- Retriever package versions are recorded in
  `environment/RUNTIME_VERSIONS.md`; the complete observed package inventory
  is `environment/retriever_pip_freeze.txt`.
- The released run used Search-R1 commit
  `598e61bd1d36895726d28a8d06b3a15bed19f5d3`; set its location with
  `AETHERSEARCH_SEARCH_R1_ROOT`. `scripts/bootstrap_env.sh` intentionally does
  not put this tree on `PYTHONPATH` because it can shadow the installed veRL.
- Retriever server used by this project is vendored as `runtime_assets/retriever/hybrid_retrieval_server.py`.
