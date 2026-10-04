# Code Inventory

This inventory covers every non-ignored file in the AetherSearch GitHub
release. Model weights, optimizer checkpoints, eval result bundles, report
archives, and runtime snapshots are not included.

Generated UTC: `2026-10-04T08:20:57Z`

## Summary

- `assets`: `11` files, `842971` bytes
- `configs`: `24` files, `45955` bytes
- `docs`: `8` files, `165575` bytes
- `documentation`: `4` files, `58811` bytes
- `dpo`: `8` files, `96078` bytes
- `environment`: `6` files, `15223` bytes
- `recipes`: `2` files, `5883` bytes
- `repo_root`: `3` files, `1683` bytes
- `runtime_assets`: `3` files, `29691` bytes
- `scripts`: `18` files, `76227` bytes
- `sft`: `46` files, `657404` bytes
- `src`: `89` files, `1290396` bytes
- `tests`: `57` files, `503940` bytes
- `third_party`: `5` files, `126026` bytes

## Files

### assets

| path | bytes | sha256 |
|---|---:|---|
| `assets/README.md` | 1552 | `bd72b7f77fb29e920050d02d1085a30b4eec9a53ef7e0676eefe6e1bde9f60b6` |
| `assets/aethersearch-mark.svg` | 824 | `5e3e0105c3db839d15b775155f8c19e02a5982d9ad01c9c57b0e4a9bfbd0f467` |
| `assets/aethersearch-method.svg` | 6866 | `803d8e2453e26b358fba5807310deca9f808d2399ca4ac5569027a79bbd020e2` |
| `assets/figures/over-search-conditional-correction.png` | 179104 | `26913420ffdff6f8cb979f9c5ba25184f239f3be4a3beca8efa5bfc59c58a0fb` |
| `assets/figures/over-search-conditional-correction.svg` | 23237 | `e38be54e2215bf5cca624227b71948f9a3053fdd0f76e7a470f896913a45010b` |
| `assets/figures/search-credit-boundaries.png` | 249479 | `9013949ae91c56ae616966126bbdf21d9f7a9a8d2fdde2cedddbf18af679152b` |
| `assets/figures/search-credit-boundaries.svg` | 43150 | `b010c913d560ad67457bc635e8efb70b54c5f4e04446bf27d275745b7ea56fb9` |
| `assets/figures/search-credit-delayed-return.png` | 137935 | `ff3363b25dc4b8db041c014f6294ee2548750854edc7ed4bf77a76c16dfdd4e7` |
| `assets/figures/search-credit-delayed-return.svg` | 18273 | `4b0a5e94f3e8db0ce7058a27e1f85f0b041695106d4041e7fa75fea69c65b87c` |
| `assets/figures/search-credit-same-depth.png` | 157219 | `45ef99e40996e8294db263311e33a04aa3d92e800c3598eaa29909d1577e7ed2` |
| `assets/figures/search-credit-same-depth.svg` | 25332 | `449750afa2c720d6b03b73311886d30f67d1a1d2996e358403de278620d188cc` |

### configs

| path | bytes | sha256 |
|---|---:|---|
| `configs/README.md` | 5229 | `546c827178ec53274daf2044b004e60f58586bf0bb29fae7d17bfc3159465b03` |
| `configs/assets/aethersearch_release_v1.yaml` | 4018 | `d4ba2aea481cf642d77abf1714bbb12a0ea7fe1eb7076c192f7b4c04dbfb1d4f` |
| `configs/base.yaml` | 6335 | `4c725cfa92a048e1d772cb0db04771b91d0f7076190fefeea93c3ad22a47c2be` |
| `configs/exact_ig.yaml` | 4274 | `8c8c7df092024d772afc4684c98158b54cd1ef2639236c1a2a284afcd50d8351` |
| `configs/exact_ig_fast_path_audit_status.json` | 1107 | `de7b1386e33e9743f930cb398c39e6445055fda1a95cddc761e3d9bc184ff8c8` |
| `configs/forced_refill_96_test.yaml` | 84 | `41da07ebaf0143c99c123cc4f63dcba01fbbda5d1969402c6b0ab985f4b97427` |
| `configs/formal_resume_u20_3rank.yaml` | 436 | `a86e58cef9cf5f890753ec6c5208ec95246c01d49010f023c21d6ba0e3ed3445` |
| `configs/formal_resume_u20_3rank_48cpu.yaml` | 681 | `8ef0bf99193d25f49797dae54853aa3500f0b7819ac62ffc32d30aa4df583949` |
| `configs/formal_resume_u20_to_u500.yaml` | 1618 | `3b481cff0555b3e68914d00cebbfbce116c2ddb1e59274ebbe838dc301652bac` |
| `configs/formal_train.yaml` | 3257 | `ca02399d3548eb3211ba28e3ab363fc3771699c5a164b21b0a1d1a2223ca3d8a` |
| `configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml` | 1850 | `158994061cb49f31453164f6cffbc65cb4ea6bd99607fe615646c501fc29e783` |
| `configs/formal_train_answer_only_ragen2_paper_mica_ig_v1.yaml` | 233 | `96ce72b40d0bccbb6a8e5683d55db8fd78ddf0500958c88876e0f7cd17dca925` |
| `configs/formal_train_role_localized_gate.yaml` | 1404 | `77f0ce4dee74c0db2aab35bc231aefb604dc9d0d0740dbf4fad75cee0a3f1a84` |
| `configs/gate_calibration_role_localized.yaml` | 1407 | `24fa289f30a1a69d2e728d081058fde996d59e37bf2cd4afa9d6b1497dc16481` |
| `configs/hardware/4x48gb_3rl.yaml` | 748 | `11aa3ae1437984407608f63d4bfe51c7c987580504313cd0fcaa9bf0b2c73983` |
| `configs/hardware_5x48gb.yaml` | 2300 | `61dfd6ebaeaceb016bf6186e652b26b7d295d5d7adbe00408d8140965305b154` |
| `configs/logging.yaml` | 1389 | `59e27ba62355340a12bc0cb5fce372b5cb42f8722aa9d6a339c6b28f8e6d26a5` |
| `configs/pilot_20_final.yaml` | 4239 | `f6ccf9555b8351810c99a2c08e6f689f12bcdbd4977834211ec69d13aceeda8c` |
| `configs/qualification/official_4x48gb_v1.yaml` | 458 | `95d79ca7261bc8aa3d2759824313c33a217451b07d677cf2e6df0b1d9524ded6` |
| `configs/retriever_external.yaml` | 607 | `9217abc9742aaac305af59b3818818ac8d74c2df3261fa7b505cf8f0ae5be87d` |
| `configs/runtime/verl_fsdp2_vllm_4x48_reference.yaml` | 2441 | `8414251c0f3460522213fe4fac5366c7996dfa889a06b08d3c59dba5b171c490` |
| `configs/update_stages.yaml` | 1194 | `5a4c9603e68898708d1d4ed82912ee9892ab3c584e8693842437c684f65ad580` |
| `configs/verl_agent_loop.yaml` | 305 | `f352dfe4704acd938871946ae01678d72cfe92907ab2c31a7ee42dd55b8e3734` |
| `configs/verl_agent_loop_role_localized_gate.yaml` | 341 | `c7377ae6e30859cb5ff1cb9e5481dc79317fd795d4026aaddf0887172b3fa27c` |

### docs

| path | bytes | sha256 |
|---|---:|---|
| `docs/data/over-search-conditional-examples.csv` | 2542 | `f59b2b84b69dda908f76f6c6aa9980b096dff3579f7b00af923cebf23991f40f` |
| `docs/data/search-credit-u325-peers.csv` | 12343 | `6a40909b5ecac065d5409716a57c1ab28b3903ec9a8cf8298f2bf02a1aad8d77` |
| `docs/data/search-credit-u325-trajectories.json` | 85428 | `def76c9035cd64b443d65914911c9b760eeff0e3fb2ca804367e3ff161d229fb` |
| `docs/over-search-improvement.md` | 18878 | `8e650cba0e4f483431ac4e6a640d701e074f3c1a4cd5436cbaefc31f636e173a` |
| `docs/rl-dpo-alternating-improvement.md` | 11426 | `942c02c758daea2fc4a87790fca00ebae84637f55b42a2b39c8a2462d5c9e068` |
| `docs/scripts/build_over_search_illustration.py` | 4564 | `899b312365ae4481d1676dec3165b3b1d95ee9bdfc72339c90fa39077d72c83e` |
| `docs/scripts/build_search_credit_figures.py` | 10966 | `57203f19203c65612e437210f5de3397cdbe60976dd0e1184a993324c609b2c9` |
| `docs/search-credit-case-study.md` | 19428 | `6dbbd0011c9936e393aa7c2452e2e7e302b540e2cd0ff34d2dfddf062b1a458b` |

### documentation

| path | bytes | sha256 |
|---|---:|---|
| `EXTERNAL_ASSETS.md` | 7853 | `8869d1dc5bd8cfc00991a376b47be389035b5c6401127bcd18bb5c9022403a7a` |
| `README.md` | 41409 | `60a3f462847aeb3823184ef8bfb8701f1813859e419b33965d6bca992f198f70` |
| `THIRD_PARTY_NOTICES.md` | 6098 | `f5eb49726de47e3ad7b3897840482d5aee7c854fc1921465e77cf8ff03dae682` |
| `TRAINING_REPRODUCTION.md` | 3451 | `0c90f5a5d676e3d5997e6b946b9994863b28a0d5483509c27345c64ac88666fb` |

### dpo

| path | bytes | sha256 |
|---|---:|---|
| `dpo/ATTRIBUTION.md` | 2720 | `e5b2efa91817590bf764a5eb753b3728ca9599a74d6f5778bc65c1f356abae3b` |
| `dpo/README.md` | 31663 | `f7745f0babe9748a44b6f2b66a80ba465f8438b5794a9d96438c35024d17c167` |
| `dpo/checksums.sha256` | 319 | `9dd7faa5bd8a7289a902fe952b84f2278aa2cbc0e602c890e1eab8d287122828` |
| `dpo/configs/ds_zero3_bf16.json` | 569 | `41e04c1a169b122eb058f1018edba49648f8c1ebd8f732751c279e10d0d981f9` |
| `dpo/dataset_manifest.json` | 2529 | `0ee034122d6ad3b88580ed3923658f4e1cc3aaa53504b7dce3c3483cc8a8f676` |
| `dpo/requirements.txt` | 152 | `f4192c805bbefd20f901cdf4fc6861486b54d158fd2123551f60d7078a4c6c32` |
| `dpo/scripts/run_train_dpo_zero3.sh` | 8605 | `abd0fdb5bfac1903a473f758a5fef3d66c1f796d305df3ef2c02391090dbaa6f` |
| `dpo/scripts/train_dpo.py` | 49521 | `a128620f562acf5833376bcadf19a7ab9f37a11e68557b0a4fbb4bb9c37d78b7` |

### environment

| path | bytes | sha256 |
|---|---:|---|
| `environment/README.md` | 2898 | `a5dcd97e6f95a8ca392514f71a8ab88f55a61e484fcd27d73ef0629bfe18fbed` |
| `environment/RUNTIME_VERSIONS.md` | 522 | `e52951f548ed1f590b7bc391b07cc5243ae5a9a7708cbe69c5bac811cc4b2d97` |
| `environment/env.template.sh` | 3302 | `8714c42a17676bb0cdb2622bb74606815c20f2037e3389050be69bde237bcb84` |
| `environment/requirements-core-observed.txt` | 388 | `e1b7c50395c6326deea04daefea70257a19370235c477559952f425ea96c4338` |
| `environment/retriever_pip_freeze.txt` | 3728 | `f077f160bb383b3dda5ebcdaf2526b511a6b0ee0612ca6510f9660cd7a5cd397` |
| `environment/rl_pip_freeze.txt` | 4385 | `2ec3728256325ed4723e0f2948888192bacc6cc140fe03c4a4b4e15453788fc8` |

### recipes

| path | bytes | sha256 |
|---|---:|---|
| `recipes/rl/README.md` | 3917 | `acfa45046606a3a60e50bb7b5246f479db40d1fbad6670d5bcb6718551bbe920` |
| `recipes/rl/train_4x48gb.yaml` | 1966 | `fa6a1372450a929b10e14168130ac19f7e5002972a6237924db1a8ec12d01358` |

### repo_root

| path | bytes | sha256 |
|---|---:|---|
| `.gitattributes` | 220 | `3a62a151b7887ee92f18a8813b4dd7257ae8864d820910233fce1d6b799b6a88` |
| `.gitignore` | 764 | `23f24c773b78c7ccefb93423edc7f8374d23748292bf984ad16cfaeffe70e499` |
| `pyproject.toml` | 699 | `9d56ccd76f2c5b6ec4ea56022340c87fe070e3ed9ba02801ed2ec9df48efb9cd` |

### runtime_assets

| path | bytes | sha256 |
|---|---:|---|
| `runtime_assets/retriever/README.md` | 1104 | `a50c8f9be51c8f0d979783d018de9cfcd7adc8ed90107f057e221974c696bad5` |
| `runtime_assets/retriever/hybrid_retrieval_server.py` | 27862 | `e04a6cc3fe2b90fe58049d703419c1c7759136f438b42cf2b5cb866e8adebb72` |
| `runtime_assets/retriever/retriever.yaml` | 725 | `151c3d85d30ca52b68a98b74ae86f5bbf99dc9c16c48e71b7474e40584190c22` |

### scripts

| path | bytes | sha256 |
|---|---:|---|
| `scripts/README.md` | 2038 | `94fdf8ca9969b3852b3082e7a452c36eae53c7b7fbf6ae5758c2f9362c40880d` |
| `scripts/_run_runtime_job.sh` | 6680 | `e2992460e9231587935ab3ebb4401fe57f83f94749ad739b5100361342930107` |
| `scripts/async_eval_worker.sh` | 1016 | `aeec7cb7da9fd8a9289560e911e01acaac8b1830521c949ac518a5a4e326d6e1` |
| `scripts/bootstrap_env.sh` | 947 | `d1e17970e9ed9746625d67d207115e99ecce8281cfdbc905d82ccfaf9bc07376` |
| `scripts/build_code_inventory.py` | 3069 | `fbb2658bbddbc8191975156ce5253ac1985f92539a719412f90631f1c8343a1d` |
| `scripts/launch_retriever.sh` | 2078 | `b5c7e1a8c413021de4638cc9a1f31c498639a7e9347f08a12f80d37a451c0112` |
| `scripts/preflight_mica_formal.py` | 9748 | `aab352af4984d9764a66f4b92bcc9f77f957eeced294b8cf919cdcf01316ac3c` |
| `scripts/prepare_verified_resume_recovery.py` | 13841 | `3a815e65163dfeb8ab6525806720118c5ea26f378e4d12a0dac979211917063c` |
| `scripts/resolve_mica_formal_config.py` | 2250 | `537457b617ddc0d1a2f40aaf814b7f46a4cdbcf7a6fed7b7ea31c84e7fd1173f` |
| `scripts/resume_rl.sh` | 176 | `6bb3b7b6f3f15167222d0aff0bb73efeb45e32dcc04ef21631ea549e145067ab` |
| `scripts/resume_verified_formal_checkpoint.sh` | 8449 | `99dadb67a296e7e33799d238cec1dd8b14745ba4b967797434a012fe16fd61d9` |
| `scripts/runtime_guard.py` | 5467 | `36ee203cf22ecb33b8560b268f9add0a475981b7dc01117bf8f2c019bf85b529` |
| `scripts/test_code.sh` | 680 | `755f7abd8560e8567685722fbb3d9e98045e6e27f8ae43bbfbff1b38df2a4e4f` |
| `scripts/train_rl.sh` | 2473 | `0ac0c8e3a75ee49b66ee16e017d067c0b9808b1663d06e86d43ad41ce47ac397` |
| `scripts/validate_48cpu_resource_profile.py` | 4712 | `73d6f24edab2f69498106642a8658e7f8c781b9eb360e205fc36598aa3369f59` |
| `scripts/validate_readme.py` | 8357 | `f5aeb20234240b6f939ca1b680ff5b1fccb42fa02b91a4ac9efc58351ef4417e` |
| `scripts/validate_static.sh` | 1587 | `4e4957917bd285aa4e140649d93cc35ecd9483170e2b2f46a8c78176e3b81c53` |
| `scripts/verify_checkpoint_readonly.py` | 2659 | `c8af6683b6f6e09e96fa91c16cf679706b4673d947e5437428239d542e6ad9d2` |

### sft

| path | bytes | sha256 |
|---|---:|---|
| `sft/ATTRIBUTION.md` | 3643 | `1f690b238f77cc3ff112a8bb8404a5fe5806d20128d9214d40743857299fc038` |
| `sft/README.md` | 17786 | `21951d5af9b5ae162c8a1f5e60d516e40afafa9b42792926bf1bc4cbcd5eebc8` |
| `sft/checksums.sha256` | 424 | `5c8428bfc4ee9a05efb31cb3e29cf969a7c6c4671f1eef7c25b3cd40461558af` |
| `sft/configs/ds_zero3_bf16.json` | 569 | `41e04c1a169b122eb058f1018edba49648f8c1ebd8f732751c279e10d0d981f9` |
| `sft/data_generation/run_teacher_rollout.sh` | 1906 | `0b4af6b685ed6ddcd0df4196f0d40d68742f8fac8b37822d7cd35c06eee7af54` |
| `sft/data_generation/search_sft_hybrid_v1/build_bm25_index.py` | 9938 | `789a3a8df399c73e8d85346f7fc683c8a154301323684f8bd6a56bad6482f17f` |
| `sft/data_generation/search_sft_hybrid_v1/hybrid_retriever_v1.py` | 18419 | `a5ebc811807bae6a697688454ce37f63796c6477e4fa635598ea71fcc0c6c775` |
| `sft/data_generation/search_sft_teacher/DATASET_PROVENANCE.md` | 4087 | `d1cdebd6e45a10c020dd94cf4b3b5a2c64d75fa497a4af19e6df109b78cbb6a7` |
| `sft/data_generation/search_sft_teacher/README.md` | 69612 | `c5ede6dd88a273e48f3bbc1be54bea48fa8bda1c4c1d82aeeb802169b8737e0a` |
| `sft/data_generation/search_sft_teacher/answer_utils.py` | 4716 | `fd025c137fc60ba8567af789ef614d57ed51b0cd1e5b58c5b0a54b0bfb70302a` |
| `sft/data_generation/search_sft_teacher/batched_retriever.py` | 3725 | `d726dd5772a13357731f4b1e4edb4aa6d5fabbe08e65d2ed079d48ecddb5fc1c` |
| `sft/data_generation/search_sft_teacher/build_sft_2600_release.py` | 16660 | `2d3d677a8eee67941bfe097ad07cd5394a9fbb657bb9791b016b64ce95748db6` |
| `sft/data_generation/search_sft_teacher/controlled_rollout.py` | 30502 | `b38bf83b7dbd99a94688abe29e0c4d91cdd70fed53845cd6c2704661f49bb109` |
| `sft/data_generation/search_sft_teacher/deepseek_client.py` | 26111 | `ff68456926c18ec0cbbd51cd0ee7645dbdf5c947df7a3b1fc1c1eac2e9946598` |
| `sft/data_generation/search_sft_teacher/deepseek_key.py` | 10183 | `91bd6add471f016647cb79f13fe706232196d637757817a511cabfd3c36f3a71` |
| `sft/data_generation/search_sft_teacher/deepseek_rollout.py` | 43761 | `7e42ad999a2064c0f660b68fecb7c25c8ba78d768848623a6c8e223bf9e916a0` |
| `sft/data_generation/search_sft_teacher/generate_direct_answer_sft.py` | 32881 | `23fde5796a6a874128c239dc1cd39459df6da820b5d43a31ef4907f9932c8d49` |
| `sft/data_generation/search_sft_teacher/probe_chat_continuation.py` | 5555 | `b7564604eec4ded56bb3f9a7d30289e9de3b4766c3d10c159c9f13391bbf9d3c` |
| `sft/data_generation/search_sft_teacher/published_sft_format.py` | 4457 | `71475b3d6c918ddab35210f9a0370f1044d82e32ce3de483013db1a9be42f95d` |
| `sft/data_generation/search_sft_teacher/replay_api_responses.py` | 7402 | `198559df30674a571fbb3894c43ffe2482916767c642c593e43649111d529bf1` |
| `sft/data_generation/search_sft_teacher/requirements_direct_answer.txt` | 48 | `0d9e4b5be3f09851fe111343e3daea45e7fec9ce0723a8084a8457a4b71a7b6d` |
| `sft/data_generation/search_sft_teacher/test_api_20_questions.py` | 15586 | `e84a7690243296e44805b8a8acb56903b8113631425c0da257371e44ac3e5120` |
| `sft/data_generation/search_sft_teacher/test_api_question_selection.py` | 4566 | `e8adfb04aac2dbe2e88e3ce43390eaba680facfa837ab1e8468a3ecdbcec3d90` |
| `sft/data_generation/search_sft_teacher/test_batched_retriever.py` | 4751 | `c52e0a42ed8e93ed7d35d9942798461bf8b044ac20f6f96df82c107a33b5d37d` |
| `sft/data_generation/search_sft_teacher/test_chat_continuation.py` | 11692 | `63f96c89e5eb3334876a193e2e1f5ac6451c9d7b5dcf0a6c7639b856f05e75c3` |
| `sft/data_generation/search_sft_teacher/test_controlled_rollout.py` | 5941 | `5ec6fb3410fd57c7f0ca27c8710a2552e900d6d1fe04ab75f37d9687800b1b18` |
| `sft/data_generation/search_sft_teacher/test_deepseek_key.py` | 7192 | `d053101f7c93b50fd520d43930366bbdf7e1eb9b572996b6eb361e223f7956fc` |
| `sft/data_generation/search_sft_teacher/test_deepseek_rollout.py` | 56328 | `56cd72623789dbece9cf4c31dcd27e2613d94dd62031e13f6c2d9fc3a127de82` |
| `sft/data_generation/search_sft_teacher/test_direct_answer_generation.py` | 3229 | `aa87af9383c18fbf936b9d4091c77dc3b1af34629d66feb1026588579186c3c6` |
| `sft/data_generation/search_sft_teacher/test_final_protocol.py` | 8886 | `6e2e83bd6a5e0ef759e0d015aab53baf7e334aed69346cdf8cc5eb299433cc35` |
| `sft/data_generation/search_sft_teacher/test_max_thinking.py` | 2188 | `ad948ca000cc6b0cb8e7b84b570715d518585ad5ef4492c787ed9e874bfd0e61` |
| `sft/data_generation/search_sft_teacher/test_multiturn_context_api.py` | 33250 | `25842bb9991838f235eda81c26f7e4322236e4e7133ca64ee73604f35df939d1` |
| `sft/data_generation/search_sft_teacher/test_multiturn_context_selection.py` | 12669 | `2a4f663e76c8fd2424a7e77899f3ee0c367576aa73259bb8fbc30d0d83488e9e` |
| `sft/data_generation/search_sft_teacher/test_published_sft_format.py` | 10416 | `fd1b8ec4472b7088def0a8abe703c6bff5a18ad2413bd3d8831b99ab0f6c7d28` |
| `sft/data_generation/search_sft_teacher/test_search_extraction.py` | 13082 | `1426bfe541272073674da037eb842ddd5a2ef140758a4a9dbc25548edee4645d` |
| `sft/data_generation/search_sft_teacher/test_search_protocol.py` | 29725 | `7d17c5f6547794374373387abc41cac01ddbc640c030f9922fbb53e0a6d214a2` |
| `sft/data_generation/search_sft_teacher/test_short_answers.py` | 9317 | `3709e9d4e646f2147b708713e056f808c5c70e529cc62cc174cf53559fa38418` |
| `sft/data_generation/search_sft_teacher/test_token_budget.py` | 6633 | `94712f8f708797a63864095921b4e713206d2c7ef3104126e33769cba3297650` |
| `sft/data_generation/search_sft_teacher/test_tool_boundaries.py` | 16097 | `aa61e54296fde43b10c7941f9f6805bf145b554cfceac36eb9160705709cfa9a` |
| `sft/data_generation/search_sft_teacher/token_budget.py` | 5385 | `5fc01417e49211133c9d8980f6a7486a9a5e84f80d49ae189f34a5c5cf572396` |
| `sft/data_generation/search_sft_teacher/validate_direct_answer_sft.py` | 4434 | `4a777cf1c2db4ccdd5e66664b174921c0c041f8a50ccf027c78e4267351e5407` |
| `sft/data_generation/search_sft_teacher/validate_teacher_rollout.py` | 29675 | `8989d62dfccce851259f20d7e3ba56d0b70696bd3aa99e00c9874297ccab041f` |
| `sft/dataset_manifest.json` | 2865 | `f76d0ada754bbd4056a67c196adebbf34fd46b9d2cc4f45ee1df1bd64d6d438b` |
| `sft/requirements.txt` | 152 | `f4192c805bbefd20f901cdf4fc6861486b54d158fd2123551f60d7078a4c6c32` |
| `sft/scripts/run_train_sft_2600_zero3.sh` | 8303 | `848a117ec64302ee427462f2299bb4ba336bcce24bb20e3c4c96a1c1634135a9` |
| `sft/scripts/train_sft_2600.py` | 42657 | `96872540c064128a6fa67076e17dda9692a3dd1f23f402862b14bffd9d7e7cc8` |

### src

| path | bytes | sha256 |
|---|---:|---|
| `src/agentic_rl/README.md` | 1337 | `9ca8ec768c5651553efdc88566090a1a36717c536b28adcbd91675e0bae45122` |
| `src/agentic_rl/__init__.py` | 78 | `5e9dabeb9b9ec3e1a11cb68b13b73f1f11d514a3dbda94ef7d0a835095371737` |
| `src/agentic_rl/advantage/__init__.py` | 1964 | `2e38f447a0511bc1bbebafa49f994eba2e814bc921aedc55ccf113d5ab23295b` |
| `src/agentic_rl/advantage/a2tgpo.py` | 73550 | `afa94ba2e083b9d1dff4707da3be941b854b2e46fca95769b7ff049774050a98` |
| `src/agentic_rl/advantage/mica_ig.py` | 15595 | `da2c3fe48716ede2355368136b7a995e9c97d50ac8b63ab6aa7bec813a460cb5` |
| `src/agentic_rl/advantage/role_localized_gate.py` | 14335 | `4faa97b10be764a94b118a9948a65918f47b9fa860d2a902180f9f7825f7146b` |
| `src/agentic_rl/advantage/stop_continue.py` | 15470 | `a9d281ce55bde5357ebbd8b1965df85c857cbec07ca4ca4f78f7b66655bf5902` |
| `src/agentic_rl/assets.py` | 4733 | `1256a791f83f4b46c600a977bbcf58ade43e9129e5ccce08095bc9e9507653d7` |
| `src/agentic_rl/checkpoint/__init__.py` | 277 | `1de6ba58c69f24283360d5b26175b2c166192462ef8c051c677684cab893e0de` |
| `src/agentic_rl/checkpoint/atomic_commit.py` | 8441 | `68e487e5f7a0ec53f74f26a052d3bfc480755ea38872a06d73d9f8cde582f466` |
| `src/agentic_rl/checkpoint/fsdp2_dcp.py` | 2494 | `48cd9009ab11f3a6a202c3bcd16a9aa128650f577af3a3174cf6a8a0bd4fb5a7` |
| `src/agentic_rl/checkpoint/state_schema.py` | 2307 | `8d0660762daf08cdb1ea14a7077215d86abb1456f335ac2396bb6ea2b47a2303` |
| `src/agentic_rl/config.py` | 55999 | `10f5c51a8ed9e6e98ea4019c68d597d7221b29e86966273f95aadb68761f9360` |
| `src/agentic_rl/controller/__init__.py` | 360 | `e90831d047994fb4bf3b7231d1451b92576b71da06b0bc82cb534cc3655485d6` |
| `src/agentic_rl/controller/attempt_state.py` | 2233 | `1da533ef6c2eb4c62a1a40dc971d66060d30db0a7a39ac1f1d0580ee51db6da7` |
| `src/agentic_rl/controller/dataset_view.py` | 7776 | `cb127681e3f3e53274c10bbfc199cde7818d2568bb15d5f34955b1401e6225d4` |
| `src/agentic_rl/controller/prompt_sampler.py` | 2845 | `d3723b8b915ac12f255889b3ec0057ec0767fd37d6a31671a79d47ccae0760e2` |
| `src/agentic_rl/controller/snapshot.py` | 3310 | `7e5151a6fc237fcca3c6b320aa43b95d372f63a345ad6954346c28cd3a1af29d` |
| `src/agentic_rl/controller/transaction.py` | 9977 | `7a5fac6428898c31895f8228566faab799da9f75ea53a1ae5dc5a270e9f6c5eb` |
| `src/agentic_rl/controller/update_controller.py` | 16214 | `dbe1733fd424a8790ae97974fb7911f2376f45cf5ad4a9700aa420f6a73b60ef` |
| `src/agentic_rl/exact_ig/__init__.py` | 1063 | `d9b16f3dd9078b2b659cf517954f99f8650b3d4364836853aeec67344e22fbb2` |
| `src/agentic_rl/exact_ig/alias_reduce.py` | 792 | `c1cddef20c970a89adebc5b437b87a525bfe2864477cf7fde8a05f9dfd2ce376` |
| `src/agentic_rl/exact_ig/fsdp_scoring_window.py` | 6718 | `c799d427b28bec90b75781e2092d738f50d80780ebb734996b90d90fe8f75b47` |
| `src/agentic_rl/exact_ig/masks.py` | 2323 | `2253e0818c5f0e808d0d9aaedd75b6ba1fa23e1f14e942fbdf32826c8e474660` |
| `src/agentic_rl/exact_ig/position_ids.py` | 1839 | `d10286645870a3991340e93826380cf52a7d54b3a374fb510396e77aceccb0ad` |
| `src/agentic_rl/exact_ig/precision_policy.py` | 8506 | `8ca382c11930527915d51fb3c6bfe3b5b01f7928b6d71b80bec3bbdff8b453f9` |
| `src/agentic_rl/exact_ig/sequential_oracle.py` | 14057 | `77e870b5f85d3ac9f4a20d51018f1e534b97ee79a65b8d257406acf456c07b29` |
| `src/agentic_rl/exact_ig/target_schema.py` | 15883 | `3ccc60b9ec7c120279dd4f828a66bf67de808c23bd82e466d6c59b04c032255f` |
| `src/agentic_rl/exact_ig/task_builder.py` | 24189 | `45d75f6fe2034170c1ac2cb160285a3787d207c819e78fb72bf3cddc3df17341` |
| `src/agentic_rl/exact_ig/vectorized_scorer.py` | 46524 | `1954d11b8e997702f2eac6f197cf4f9ee44d035c66dd902876d453d199d379de` |
| `src/agentic_rl/metrics/__init__.py` | 230 | `cac923cb267bb414453b20ebd5c7a894d045043b6ded27dbf7a55af1710d2473` |
| `src/agentic_rl/metrics/reasoning_collapse.py` | 1974 | `2d99606ff578c90c7e61a1dacda9581beec1ff11b4696bcfdd3557d5d1aab31f` |
| `src/agentic_rl/metrics/runtime_records.py` | 37508 | `4396f610a358de9f4bb9475f961c5270e1c72e6a6f96ba2e0177cc7a1b8b78c3` |
| `src/agentic_rl/metrics/schema.py` | 5483 | `c83bc12de48616633cd03067e3b3b2f155ca5fd80d0bdc6ed5e3f1df022b6fb9` |
| `src/agentic_rl/metrics/sinks.py` | 1651 | `245be3b3c9e9524cfdd606d39f4c602745c1e30fcc9f424194d7341410edb9b7` |
| `src/agentic_rl/outcome/__init__.py` | 609 | `bfe824e889021cb018aa105c44ed0fd0ec0b1584bfa9cfb123d3e352fe174b99` |
| `src/agentic_rl/outcome/format_indicator.py` | 504 | `896e32e5598a5d6da8cadff84c011ac5ff9df7ca577715b7b2243867e709c4fa` |
| `src/agentic_rl/outcome/parser.py` | 6456 | `8cf32c38c147728e4a1da0059cd53d04cd3178cbbf29301823e4b1f1b9fa320b` |
| `src/agentic_rl/outcome/token_f1.py` | 4635 | `f9d251dd402607091fd606cbfdf9a2c0abd6d42fd368549eac8e33800cf9f54a` |
| `src/agentic_rl/outcome/workers.py` | 3976 | `a82df1f0bf2a96e865ec4612949690fc1fa996f8532235890521691542530bc8` |
| `src/agentic_rl/policy/__init__.py` | 236 | `f12721ebe1ad5bb903bc4ba42fa5a106127d4f3640807fb3b9a13e3723a62eb1` |
| `src/agentic_rl/policy/gate_gradient_calibration.py` | 12837 | `93fb5569ff7e6257de767559ea81a2493f5158bf5690182cfadc952d724ecd7c` |
| `src/agentic_rl/policy/reduction.py` | 3178 | `d8b373907e9cf638cfc8205e6eb8fe909588b9fbc9b0abdf9263402990a11a77` |
| `src/agentic_rl/policy/reference_kl.py` | 6835 | `d2353c5217ee27fb08357cb8f294201344974f64b52194bd21c90506500bc4e3` |
| `src/agentic_rl/policy/strict_onpolicy_loss.py` | 8120 | `7852885cecad09cb815dac2fac06653eefe145cd154c97c0546c9b9867100284` |
| `src/agentic_rl/policy/turn_ratio.py` | 1737 | `1e1c54e3931d2a19afe706f29c508a995fd455a46002bcfdcf054e97242a8e34` |
| `src/agentic_rl/qualification.py` | 5685 | `168d7e3f26aafa5aaec779415b533da2fb27fd49eb9a0edd5bd168580121dcaf` |
| `src/agentic_rl/retriever/__init__.py` | 403 | `6204c8b285e230615df23d837d37055a9996a3d5ce7ec26675056ebd355687ec` |
| `src/agentic_rl/retriever/client.py` | 10898 | `8655687dcfd14001bb11bc2bb629cba6f90f1d80ebb30562773cf95bccd1a410` |
| `src/agentic_rl/retriever/health.py` | 1631 | `806af23e4fd8bcee1796223fe1193844d2f938abb705f5304c5f51c651999ce8` |
| `src/agentic_rl/retriever/protocol.py` | 1982 | `cc40c8e887f4e301c2937fbec6176e1db135c18ca3a8166009ebc7f1568c7705` |
| `src/agentic_rl/rollout/__init__.py` | 320 | `93f69bb07acb4304152798e0a6555ef46c2d8cafb9a058c0a70574dba3a0b8b8` |
| `src/agentic_rl/rollout/agent_loop.py` | 3093 | `6eea80d1f931d07159e7d24d83848b6561de4041a1458ade4a7b7bcbf0e06631` |
| `src/agentic_rl/rollout/search_role_provenance.py` | 19574 | `ce811c665b914a57ce308e16ad311681c7c35daa13c1456329e56dbdd497ef7b` |
| `src/agentic_rl/rollout/token_provenance.py` | 3381 | `7b253a5881c9a0beb1343598d5e94f4a5bdedf63c2679e3993414f4bddb826b1` |
| `src/agentic_rl/rollout/trajectory_schema.py` | 25751 | `885677df7d0ac2e7b0341594f23b9baccb55a12204425f6e0ccd745ff61e34c3` |
| `src/agentic_rl/rollout/vllm_manager.py` | 2624 | `497a365e83eae4ffb48319db313da983ccd7c1e72d918238c6927f001f230ee9` |
| `src/agentic_rl/runtime/__init__.py` | 725 | `831726e72cf248b236926bddd60917d4302927f6bf42a06fe1b1fd3be017d7bf` |
| `src/agentic_rl/runtime/async_eval_worker.py` | 19809 | `2def30522f04b4bee7dc18821cc392e52dd39e1959cc4ebfa70cb5e2e4a2b546` |
| `src/agentic_rl/runtime/capped_vllm.py` | 29582 | `07d33a34c5dea7725216cc1fb660e26a7cf6c594d7fa7a4c269d3d216d4cfee2` |
| `src/agentic_rl/runtime/entrypoint.py` | 491 | `b89c28df1a5f9ac848cb3b77487efe614d680b1f908a5ffc6d0bc4b0347bb68d` |
| `src/agentic_rl/runtime/environment.py` | 3411 | `0d2fe6945210566e6518261f5d2aa2934796e464fd37f2545ffbf35d3e681d15` |
| `src/agentic_rl/runtime/fixed_eval.py` | 8457 | `38de064da6592d051d3ae368b2088c97ad617f0599e2819a60db8f51f68343bd` |
| `src/agentic_rl/runtime/formal_state.py` | 6516 | `0b3ed21fff94fa9274074eb3097ebfe80116213b518d049533d32728f7368c37` |
| `src/agentic_rl/runtime/fsdp_worker.py` | 122408 | `25f7e1a47762ac022e66caffe7122a80487d124169a1e9b0a14a15b7a1ad83c2` |
| `src/agentic_rl/runtime/learner_batch.py` | 22565 | `6f03c822e9c9f6253cebe03d44131aca014c4304be39102a9f29456da4cb333a` |
| `src/agentic_rl/runtime/postprocess.py` | 11336 | `d7eaece236a674e227290fe93967968ab041fa532d64543bf46f4e8338d888b8` |
| `src/agentic_rl/runtime/pretrain_controls.py` | 3292 | `4e61e6cf5478035e32891b57304350ec8cb16ba54f1dec20bfb9bca33e56c17b` |
| `src/agentic_rl/runtime/ray_topology.py` | 17310 | `f676823118f311545b9a70a09b8523b150a6ba2a77512dd35070ff509e0aa435` |
| `src/agentic_rl/runtime/resource_guard.py` | 16734 | `7945c329f6ac1e7c007140e0773cc6d0570f49d066e8ed708b59b5f663052bab` |
| `src/agentic_rl/runtime/resource_plan.py` | 9388 | `8d8fda728a4b37bf71aeb4acf986aec0d6e350b0a35d58ae6ca8289155f6c1c8` |
| `src/agentic_rl/runtime/retriever_command.py` | 2995 | `ac344a92f1f8bd99ac932b85c32bd0950ccf4f046ca738560d2e24c0768dbbcf` |
| `src/agentic_rl/runtime/search_agent_loop.py` | 31982 | `231fce4e80cea00686587e53dff2695f9979d3adb159faa3827a61f38478aa47` |
| `src/agentic_rl/runtime/stop_branching.py` | 57123 | `dbc79ac154d6882b2328001a4f02d50182f56ca1f676776712331b0bc889591b` |
| `src/agentic_rl/runtime/verl_config.py` | 15882 | `fa94360f9f004d21ecdfbcd3931dd52f4c64b1e2e3da05398ef92d7f4c4f70af` |
| `src/agentic_rl/runtime/verl_runtime_adapter.py` | 310762 | `e7ea9f4d6c4bebcb790cd97da1351b6315e1b4c6280f93baa23f6313e1a02ef4` |
| `src/agentic_rl/selection/__init__.py` | 931 | `bdb9c618bd263a4ffc39a2da3f8f176ccb422c0319731cbd375caafed94165c9` |
| `src/agentic_rl/selection/candidate_pool.py` | 15734 | `c88b723b6c14fa538989017d613961231ef9d9d75d214fd6674211c3b2aab192` |
| `src/agentic_rl/selection/channel_scale.py` | 6344 | `b869bc5b269be447465e9c8a12532593bdfee45280b77d5a091dc58d450a2991` |
| `src/agentic_rl/selection/health_gate.py` | 1158 | `768ea639bee3163340a4730a913de61a343971e57912e471b33d414cba1a7649` |
| `src/agentic_rl/selection/paper_ragen2.py` | 1235 | `b61a658f62d3da244078f68b0af61df9a6fee2838ccd8450c4ca7bcdfd5e5b1e` |
| `src/agentic_rl/selection/prompt_variance.py` | 4169 | `e0709b1e969af3ee72478e1234ef886f4099f856a4a917da246c94b057cb9b2d` |
| `src/agentic_rl/selection/top_p.py` | 1782 | `65cf55c3d9c616ed87fa8f64a62d0c3cae67d0e005d806aed2127ad96ca8a6b6` |
| `src/agentic_rl/topology.py` | 15857 | `ebed408fd20fcb22fd7d61298ad92dba81b3cedda627102d4641f6551b957492` |
| `src/agentic_rl/workers/__init__.py` | 174 | `adda67db60b379cb6126fd0620076545ae5f715a5c3011b08c32054a5a28536c` |
| `src/agentic_rl/workers/failure_coordinator.py` | 1105 | `df7bfd662936632e54c4576866c2902ec3a6f1f2623922315f77d02aff5c03f6` |
| `src/agentic_rl/workers/fsdp2_engine.py` | 1718 | `bc66ff3a9bc1c4e5db18f6c8a7ac47683fe16e98b5963d98ef7839b5069dd5d6` |
| `src/agentic_rl/workers/ray_actors.py` | 15299 | `c7991b5d9df503199cedd4a3a12cbc142a5ed89da79d3aefd6c6bac6ec36f97a` |
| `src/agentic_rl/workers/resource_plan.py` | 1622 | `63e41d08841c1ac6b9a530c6e58df87bf1e847b4495f1ff271b92ac089b063e1` |

### tests

| path | bytes | sha256 |
|---|---:|---|
| `tests/README.md` | 988 | `0542e9d452605486497eb6255397b11b532450bf9470557441b0d03effb3750d` |
| `tests/config_support.py` | 1085 | `3c6bd1ca35c18be6d305fb79a429d8883f107c41e74bb3a7370a3c0c80764464` |
| `tests/fixtures/base_5x48gb.yaml` | 1173 | `89ccf62112714738d2ea6924053e9c4580242379fc3de9baa54801b832fc845a` |
| `tests/fixtures/formal_resume_u20_3rank_4x48gb.yaml` | 236 | `1c77111b8ceaf5d4a1c21fec749153c89e355ed816b4abb9f562ccd889c0e8bc` |
| `tests/fixtures/formal_resume_u20_to_u500_5x48gb.yaml` | 262 | `d86babc55847a53a3a72eb8275ade30b623490e241f679c8b52a03f151ad04d5` |
| `tests/fixtures/formal_train_5x48gb.yaml` | 303 | `8ad9660b8f28c90cc5fff5c7c1193c94e4a400d75e5e39ac46ca5aa6a6ede813` |
| `tests/fixtures/formal_train_5x48gb_base.yaml` | 154 | `e7926c4c3bf89c82db115a26ce2a78b498059527f930a9cd6714ae9ca9e17266` |
| `tests/fixtures/formal_train_mica_4x48gb.yaml` | 310 | `0245b4e653608f9c8a705f74273d8b806ec5e9871cef415e950cff319ac65255` |
| `tests/fixtures/formal_train_paper_mica_4x48gb.yaml` | 257 | `76340ccd88fe1c512a15ed6c33465ed0dc44c7f9975d9eb50aa430f136ca0910` |
| `tests/fixtures/pilot_20_5x48gb.yaml` | 236 | `9612e025af31616e6a6f54e8fa54d63b7621a006326e3c9ba5400cd4e224ffa0` |
| `tests/fixtures/reference_4x48gb_resolved.yaml` | 17394 | `5b41476b8a93243be8eccfb85d1387d930ac9ce5a0c7595ffd91a3c541039870` |
| `tests/gpu_test_guard.py` | 381 | `89e7b1d626f48868ef99e49382835c797cb2ddc3125b8daa3b7e2fa4ea85d700` |
| `tests/support/exact_ig_fast_path_audit.py` | 81283 | `f9954f9b143274cc2eb7c2b8dda891227ae3628db5da5be33b51ae21f4ede419` |
| `tests/test_48cpu_resource_profile.py` | 2770 | `8d0a14001ecd72fbc8842c713e929d8ff872282d6eedee6ef371d42376f52fc4` |
| `tests/test_a2tgpo_advantage.py` | 9401 | `67a2451d0c70e53ebf75c740506ace0649f96dca19ff5c266991a50347ec149f` |
| `tests/test_answer_only_ragen2_mica_integration.py` | 16416 | `d0bee5fa1b0a01a6fda21c216e2202f18d2cd48d1faba45fedfca61e8bd81dcb` |
| `tests/test_async_retriever_client.py` | 3026 | `319456e20f6b7764dc1094f81ad2938f042614a08e91061d83f12562b0831031` |
| `tests/test_config_schema.py` | 8095 | `bc07855bf0e8298bdd7a68b139e015e700d01737d1204d6485e6fbfa6aeff359` |
| `tests/test_dataset_view.py` | 1600 | `66a57c987800e61425d5eb30452b0baaa118f42a7925b55eeb0430d434051d54` |
| `tests/test_dpo_trainer.py` | 11512 | `f27b754a7a7fdc1678e0205f9402a900b640f6e9954bba3d7f9ba54dec14184e` |
| `tests/test_exact_ig_fast_path_independent_contract.py` | 9906 | `436a8011119e49cad3257185519975de4cfd93d66996611eef05e8fdd9fc40e8` |
| `tests/test_exact_ig_precision.py` | 7748 | `7fdda12f1cbf3e4eeef38a4f0434ecf7552beb7f5e217a51097cb7b7a89fbc34` |
| `tests/test_exact_ig_state_targets.py` | 9126 | `77c302748bdf62f70b25d3aa5906cd50fa9ac2a961cd3b4a1224b06e328ff845` |
| `tests/test_exact_ig_structure.py` | 17048 | `08d6dc513dac4d433bc4263a6ecc41809c99a0b13763a79edc643e35d048bda5` |
| `tests/test_exact_ig_v4_fp32.py` | 10101 | `031a22082c511526f83a9a3e23a8c4963d6c3c8f23906b37f674981b3301cff6` |
| `tests/test_final_asearch_production_contract.py` | 9241 | `9c4cffb34e550e89542d87789cd81f2eac19d567e4cd7b0afe6a6b70844d2fd2` |
| `tests/test_final_pretrain_controls.py` | 3716 | `1ce7002fb5cc07eb91e3c3ccb1f38dd45b9ab5cf85f82304c7e8e330ce465414` |
| `tests/test_fixed_eval_full_manifest.py` | 6230 | `d35716d78b9deaf72183901c0a659b10b97ad3346ddb2eb9f06418cca8a80ff7` |
| `tests/test_formal_resume_runtime.py` | 13271 | `764c918c9175ca408a91fe0c80b6c1ef1f0c941b3770b93d5a76334e61b0d798` |
| `tests/test_format_advantage.py` | 4810 | `efaff86d140521e142b2c003022d9ca7c16940692099f1415ae496fb00822c06` |
| `tests/test_fresh_formal_sc_launch.py` | 3342 | `4e55d3ea08329f36b4da11c288f814e1def2b38d4212a9ccaf54cf18803423ed` |
| `tests/test_igpo_f1_parity.py` | 1651 | `d2abb6cd8637921af94e1529d8dcf3c9607abb4d6777c0453a17c5d46f53f4fa` |
| `tests/test_mica_ig_v1.py` | 10622 | `66a9f40d6b553fa438955808a5ec409a4d60c09f338733d537cb1f3aec3f37ce` |
| `tests/test_paper_ragen2_selection.py` | 8900 | `e37304b2e2663743d45b91f68dff0b506211a2acc918b0f2ca78dcd7d1567c71` |
| `tests/test_policy_reduction_and_kl.py` | 7385 | `3c4b082bbfa588d6038a6016874d156cd8f8773bc550c141f1b4b643822c00c6` |
| `tests/test_prompt_sampler.py` | 450 | `06abb7b2a7f0d1d9e0050a319fb4e59f64dff2d6283c1a99f58b8511e53a3341` |
| `tests/test_public_config.py` | 828 | `69ca8c8dee34a3dbd9e3fa98ed5fb747c6a8f2a7b91ec565af36760b7fa7e381` |
| `tests/test_resource_guard.py` | 8427 | `89e1f75c2fb40455e09cfdfa3363ed153ae667812033a7714e8873d58fd024dd` |
| `tests/test_role_localized_gate.py` | 32392 | `1be5b1aa5a247bff44348bf7735c2b5bc4646abc4af49291466b81837d106e28` |
| `tests/test_runtime_adapter_static.py` | 15824 | `4dba92615f0c955649c9f96df88b43246b9b35dcc0386c5a0166b9f60b4061ed` |
| `tests/test_runtime_learner_batch.py` | 2029 | `bd5294eb8ec58176458d1e92a5c1f8ae59fc86b0f641e0b51fd7f9b447f101e6` |
| `tests/test_runtime_ownership.py` | 29023 | `4f7d2a8c3d6fa08ce8554bd4f9d7642953724d9a19fd3e070854cea1e04ee858` |
| `tests/test_runtime_preflight.py` | 548 | `83310072bbc1bea7a44e49ad6bb26fceacae1674b8de2e9c9a72ae7f2de71050` |
| `tests/test_scale_update_stages.py` | 7459 | `c62032232800cdc6c6ec98b7b9c21ebb93a147facc73e5d3d1e629fb37ec8da6` |
| `tests/test_selection_boundaries.py` | 3891 | `dcb74be9d0c099e3d0591d0b36cc9958dffd4a81b8df278da46224a9e1289098` |
| `tests/test_selection_math.py` | 6788 | `6d0423fe27153d11388c746bb19b04bb121b8016d38e8223796163ffa53bd1b0` |
| `tests/test_sft_2600_trainer.py` | 8794 | `8177f86e6b3d9fc8853dc9f0c93e6324ef0c944432bb1e650e2ea0d738da704a` |
| `tests/test_snapshot_checkpoint_contract.py` | 3789 | `c9cd0a265ef1abc6e7b56fbecf10533f3f416babedad426f2806d55861bc2143` |
| `tests/test_stop_branching.py` | 22763 | `6aeb327d4ba8e543f3043b851fefd752c7d48204cde357dbd88bbc16dcda8745` |
| `tests/test_stop_continue_advantage.py` | 13167 | `d01f9bc5e075dc558c4aeeaee6adec516ec438d652d4e5614f68c0627c8f071e` |
| `tests/test_strict_one_step_contract.py` | 5673 | `da22561fa55041a9bb1a864254811740c3ff1309a8110bc4824e43a447c60111` |
| `tests/test_sufficiency_novelty_cumulative_probe_routed.py` | 16944 | `59e8ac0f37279d2b23ace5d77ca776def6a6db016a6fef074bbb831143e226f4` |
| `tests/test_sufficiency_novelty_local_ig.py` | 9272 | `e7ffeabeae8c69cf7d9a9e7bb04218789ba634b568e131360d58b6736282ce31` |
| `tests/test_token_provenance.py` | 6861 | `cb50c8b20251305b4de64ce3972718ea0bb0544c81fe7f18ad5ded0cbf6d5421` |
| `tests/test_topology_decoupling.py` | 19233 | `9fb7451704ace0a789a60628396e73975668172372e856ecc07f3ec5fd2868ac` |
| `tests/test_update_controller.py` | 8296 | `08a9ea06361ee35d50dfa82c1b69db7c12aefd9035e27ede0529f832ee820bcc` |
| `tests/test_validate_readme.py` | 1510 | `15492ea829657ae854480c52f3ed86e86a9b49b277833b379db8580e3e660d26` |

### third_party

| path | bytes | sha256 |
|---|---:|---|
| `third_party/README.md` | 776 | `ceddc84af12f44ef4df96c188b9c9403b7f9b6a2e59d435cf8138a61940dd9a1` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/LICENSE` | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/generation.py` | 54797 | `7019243992a3b70fe4d74d3fff6808cb3d7e25fa0b9ec461a4effa41681333b0` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/prealigned_vectorized.py` | 23073 | `636edca70e84408a988bfb2ff6c7ea0747d3f2bdbc5591d0ad34e3813c963cc9` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/vectorized_gt_logprob.py` | 36023 | `a00da4b594238baa9b2fef911fb5d0a418c5c258d5097559fff3fc6389689f9d` |
