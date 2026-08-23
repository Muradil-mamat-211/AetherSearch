# Code Inventory

This inventory covers every non-ignored file in the AetherSearch GitHub
release. Model weights, optimizer checkpoints, eval result bundles, report
archives, and runtime snapshots are not included.

Generated UTC: `2026-08-23T13:14:26Z`

## Summary

- `assets`: `2` files, `1154` bytes
- `configs`: `23` files, `38388` bytes
- `documentation`: `4` files, `30881` bytes
- `environment`: `6` files, `14527` bytes
- `recipes`: `2` files, `5177` bytes
- `repo_root`: `3` files, `1441` bytes
- `runtime_assets`: `3` files, `29251` bytes
- `scripts`: `18` files, `74841` bytes
- `sft_data`: `7` files, `17455` bytes
- `src`: `86` files, `1252902` bytes
- `tests`: `52` files, `430408` bytes
- `third_party`: `5` files, `126026` bytes

## Files

### assets

| path | bytes | sha256 |
|---|---:|---|
| `assets/README.md` | 330 | `24fb84c91d614252b1bf037dc24e21eec1abd690782fc00c085a8021dfb9a996` |
| `assets/aethersearch-mark.svg` | 824 | `5e3e0105c3db839d15b775155f8c19e02a5982d9ad01c9c57b0e4a9bfbd0f467` |

### configs

| path | bytes | sha256 |
|---|---:|---|
| `configs/README.md` | 2820 | `ec4ea24c3142d1c9c71202aea8c2aa0753c03c36a2c045add6ec7d2d041da665` |
| `configs/assets/aethersearch_release_v1.yaml` | 1553 | `7172627c2dfc44c46385ccd0004955209cb5b34b057b5ba549621bb507dd20f9` |
| `configs/base.yaml` | 6335 | `4c725cfa92a048e1d772cb0db04771b91d0f7076190fefeea93c3ad22a47c2be` |
| `configs/exact_ig.yaml` | 4081 | `f62726defbd24e1a1fca8c8083c85da7481f08a312ed49e26480d6b6d6195ab0` |
| `configs/exact_ig_fast_path_audit_status.json` | 1107 | `de7b1386e33e9743f930cb398c39e6445055fda1a95cddc761e3d9bc184ff8c8` |
| `configs/forced_refill_96_test.yaml` | 84 | `41da07ebaf0143c99c123cc4f63dcba01fbbda5d1969402c6b0ab985f4b97427` |
| `configs/formal_resume_u20_3rank.yaml` | 436 | `a86e58cef9cf5f890753ec6c5208ec95246c01d49010f023c21d6ba0e3ed3445` |
| `configs/formal_resume_u20_3rank_48cpu.yaml` | 633 | `836dea34298e107a90a663ec5702fecfd57d74d4c8e1f0048e04aa62f9fd6ac1` |
| `configs/formal_resume_u20_to_u500.yaml` | 1618 | `210de2db51bde0cf27ea24a23e84b86cfacd0f370c6f356c7b3dd89f99359e74` |
| `configs/formal_train.yaml` | 3264 | `27391ddefbe3b74ad0333b781bb55a748033a6d3b86666b8ef4f71fc5da9878d` |
| `configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml` | 1850 | `158994061cb49f31453164f6cffbc65cb4ea6bd99607fe615646c501fc29e783` |
| `configs/formal_train_answer_only_ragen2_paper_mica_ig_v1.yaml` | 233 | `96ce72b40d0bccbb6a8e5683d55db8fd78ddf0500958c88876e0f7cd17dca925` |
| `configs/formal_train_role_localized_gate.yaml` | 1404 | `77f0ce4dee74c0db2aab35bc231aefb604dc9d0d0740dbf4fad75cee0a3f1a84` |
| `configs/gate_calibration_role_localized.yaml` | 1407 | `24fa289f30a1a69d2e728d081058fde996d59e37bf2cd4afa9d6b1497dc16481` |
| `configs/hardware/4x48gb_3rl.yaml` | 1414 | `89b81589bf1c3090f7db0d4f45692201c74e1a4d35ef2f3ea35ea0952801395b` |
| `configs/hardware_5x48gb.yaml` | 1076 | `c674f0a3c4f53f990a6ee44cdaae26762ea3bbf9dfab0f7dd998e53da3a2514a` |
| `configs/logging.yaml` | 1389 | `59e27ba62355340a12bc0cb5fce372b5cb42f8722aa9d6a339c6b28f8e6d26a5` |
| `configs/pilot_20_final.yaml` | 4244 | `936b5205b84a6fb68d8630070b2c0f033d45f45798f2608a5ecfaacbf8c0c2db` |
| `configs/qualification/official_4x48gb_v1.yaml` | 458 | `95d79ca7261bc8aa3d2759824313c33a217451b07d677cf2e6df0b1d9524ded6` |
| `configs/retriever_external.yaml` | 848 | `1a188b041e1b42642297b987d33ea53a02974c2216195e72034d211ad8c2acb7` |
| `configs/update_stages.yaml` | 1194 | `5a4c9603e68898708d1d4ed82912ee9892ab3c584e8693842437c684f65ad580` |
| `configs/verl_agent_loop.yaml` | 452 | `4215f59b7472b24c573f313f18d907c78aacf59bdbfab8530ad3b37d82cf59e8` |
| `configs/verl_agent_loop_role_localized_gate.yaml` | 488 | `596b89c8e7210a48c2bd8e404e0170fac201ecea4e67d93a08a23d8f96f2c8b3` |

### documentation

| path | bytes | sha256 |
|---|---:|---|
| `EXTERNAL_ASSETS.md` | 1517 | `d9c7c100f8a16071c8774c472337933d41b27a3927e0a0d909061f11aa50e756` |
| `README.md` | 24445 | `8e143f5915a9afabc40c5a295b7668ab6324c15c96148e9498da50897d47e22f` |
| `THIRD_PARTY_NOTICES.md` | 4092 | `09c4e8e0e4c97f56361c9bd97959398b21a9ff27f0497e9ca7a6cf0cefc3d605` |
| `TRAINING_REPRODUCTION.md` | 827 | `0f459c3f741801007d21d4d505039486efcfc67fd1affc71cdb179dbc81d06c2` |

### environment

| path | bytes | sha256 |
|---|---:|---|
| `environment/README.md` | 2206 | `684a58d52b67f773777b2f16fb954e5fc0caea34a7cb88132170e0f7ba4e5489` |
| `environment/RUNTIME_VERSIONS.md` | 522 | `e52951f548ed1f590b7bc391b07cc5243ae5a9a7708cbe69c5bac811cc4b2d97` |
| `environment/env.template.sh` | 3298 | `de0bf67567411d846eed7d607db5d75edebd328da3fc5f026be3a7d889d26917` |
| `environment/requirements-core-observed.txt` | 388 | `e1b7c50395c6326deea04daefea70257a19370235c477559952f425ea96c4338` |
| `environment/retriever_pip_freeze.txt` | 3728 | `f077f160bb383b3dda5ebcdaf2526b511a6b0ee0612ca6510f9660cd7a5cd397` |
| `environment/rl_pip_freeze.txt` | 4385 | `2ec3728256325ed4723e0f2948888192bacc6cc140fe03c4a4b4e15453788fc8` |

### recipes

| path | bytes | sha256 |
|---|---:|---|
| `recipes/rl/README.md` | 3266 | `6176ef5dfd69c714850bf352966eaaaeb3c7b0ec8a02495f31eda467ac1a5231` |
| `recipes/rl/train_4x48gb.yaml` | 1911 | `d780c5c6039c9cf66a443bf3311975f6b30bffa0d8a35519c484822b974a3856` |

### repo_root

| path | bytes | sha256 |
|---|---:|---|
| `.gitattributes` | 220 | `3a62a151b7887ee92f18a8813b4dd7257ae8864d820910233fce1d6b799b6a88` |
| `.gitignore` | 522 | `7e46b7f10dfc419fde3ed30ab797d192fbf4c806bd20cf1de411f236a44f87f3` |
| `pyproject.toml` | 699 | `9d56ccd76f2c5b6ec4ea56022340c87fe070e3ed9ba02801ed2ec9df48efb9cd` |

### runtime_assets

| path | bytes | sha256 |
|---|---:|---|
| `runtime_assets/retriever/README.md` | 664 | `06869ba51fb9ce756b403b5f81d5253eda3f7f967ce1b67fa559387dbd3a87b7` |
| `runtime_assets/retriever/hybrid_retrieval_server.py` | 27862 | `e04a6cc3fe2b90fe58049d703419c1c7759136f438b42cf2b5cb866e8adebb72` |
| `runtime_assets/retriever/retriever.yaml` | 725 | `151c3d85d30ca52b68a98b74ae86f5bbf99dc9c16c48e71b7474e40584190c22` |

### scripts

| path | bytes | sha256 |
|---|---:|---|
| `scripts/README.md` | 1683 | `f87088a1823d58b555f689403ccfdd33574122ead7a3c6f33da7d3b42418a88e` |
| `scripts/_run_runtime_job.sh` | 5756 | `056ac009a6bad7a1b838ad1d9691698daf37c44b1253ee6e42f835a1c6792ac5` |
| `scripts/async_eval_worker.sh` | 1016 | `aeec7cb7da9fd8a9289560e911e01acaac8b1830521c949ac518a5a4e326d6e1` |
| `scripts/bootstrap_env.sh` | 947 | `d1e17970e9ed9746625d67d207115e99ecce8281cfdbc905d82ccfaf9bc07376` |
| `scripts/build_code_inventory.py` | 3069 | `fbb2658bbddbc8191975156ce5253ac1985f92539a719412f90631f1c8343a1d` |
| `scripts/launch_retriever.sh` | 3341 | `d65bb3ac6aa80baa3a92de61f5c29b1ffd5331e122219e974ad93ced441faf8f` |
| `scripts/preflight_mica_formal.py` | 9748 | `aab352af4984d9764a66f4b92bcc9f77f957eeced294b8cf919cdcf01316ac3c` |
| `scripts/prepare_verified_resume_recovery.py` | 13841 | `3a815e65163dfeb8ab6525806720118c5ea26f378e4d12a0dac979211917063c` |
| `scripts/resolve_mica_formal_config.py` | 2250 | `537457b617ddc0d1a2f40aaf814b7f46a4cdbcf7a6fed7b7ea31c84e7fd1173f` |
| `scripts/resume_rl.sh` | 176 | `6bb3b7b6f3f15167222d0aff0bb73efeb45e32dcc04ef21631ea549e145067ab` |
| `scripts/resume_verified_formal_checkpoint.sh` | 8449 | `99dadb67a296e7e33799d238cec1dd8b14745ba4b967797434a012fe16fd61d9` |
| `scripts/runtime_guard.py` | 5467 | `36ee203cf22ecb33b8560b268f9add0a475981b7dc01117bf8f2c019bf85b529` |
| `scripts/test_code.sh` | 680 | `755f7abd8560e8567685722fbb3d9e98045e6e27f8ae43bbfbff1b38df2a4e4f` |
| `scripts/train_rl.sh` | 2473 | `0ac0c8e3a75ee49b66ee16e017d067c0b9808b1663d06e86d43ad41ce47ac397` |
| `scripts/validate_48cpu_resource_profile.py` | 4811 | `d02248f655d0ff0d2745fb21d03e089e177365cfe89d2d3678ce834da8df8823` |
| `scripts/validate_readme.py` | 7633 | `af02e194a98f45b4cb55ec11fa44dc47ee2c5de0ad1598c5f922e6048800ff4a` |
| `scripts/validate_static.sh` | 842 | `fa4a04cc7825733a8d5ce799486ecfcfe47311a1d4b9064e553d52e4b3d9d27a` |
| `scripts/verify_checkpoint_readonly.py` | 2659 | `c8af6683b6f6e09e96fa91c16cf679706b4673d947e5437428239d542e6ad9d2` |

### sft_data

| path | bytes | sha256 |
|---|---:|---|
| `sft_data/ATTRIBUTION.md` | 2776 | `11ca5354884cad2bbc64b224a930d263e741f3d0b9148490152811f9cc24822c` |
| `sft_data/README.md` | 3975 | `315d69d5f89f5c936964d26fae47fd3b912ed4df89f8e94a4a79aa0cdbc17715` |
| `sft_data/checksums.sha256` | 419 | `e6b1fc982ac115459215e718e031c5b83a66e53bb9c559b7916c7da265b58e11` |
| `sft_data/dataset_manifest.json` | 1444 | `3b6ca2879d1d680d3226ed96c326e1ee1ff566d4eabe73ed005619096e02de8e` |
| `sft_data/scripts/build_audited_queryrewrite.sh` | 4533 | `36cb1cbf7497c12a615d50f141839eee2c9944ceb22a71a911a379eb6680a566` |
| `sft_data/scripts/build_final_sft.sh` | 3054 | `913d285bf31732d0109efb7556af0eb59145ec94333a72da271311a752393a4f` |
| `sft_data/scripts/standardize_queryrewrite_think.pl` | 1254 | `4f6d97cd6cb28c241f6b5e4807f61f39abdcd628cdaa1d03e661604e65ba2ff6` |

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
| `src/agentic_rl/config.py` | 47913 | `a156950e38cc8d206900567daadfaf5dca99c33b781099a5c420b48c45311746` |
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
| `src/agentic_rl/exact_ig/sequential_oracle.py` | 12562 | `0bf3fe214c946b9704c90a34131b692a4a82967ae5236eaef8af869051ce4df1` |
| `src/agentic_rl/exact_ig/target_schema.py` | 14631 | `307b18bda63a4c3d51070a5d9d69526f1a32444a26cb13d1a202623d67ddf457` |
| `src/agentic_rl/exact_ig/task_builder.py` | 21031 | `8af56f9f02918417421b249b5d290f76b1652182c423a9f446f6db3da8d956ef` |
| `src/agentic_rl/exact_ig/vectorized_scorer.py` | 46073 | `f990917302c2857d61b37966a24844310f736b3fafd7f93c57145a9c164f6be0` |
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
| `src/agentic_rl/qualification.py` | 5596 | `3f00e2be598afc8f4cbfeb4e6944172c18b69f6136fd9878d28c12b5ac5a285d` |
| `src/agentic_rl/retriever/__init__.py` | 403 | `6204c8b285e230615df23d837d37055a9996a3d5ce7ec26675056ebd355687ec` |
| `src/agentic_rl/retriever/client.py` | 10943 | `9bd9f8429279bc02c4589c9bc23a55f9aa390385a598576f7b91932135153b85` |
| `src/agentic_rl/retriever/health.py` | 1635 | `6ee579cbd85a513f82a85fbd8dfe2423a65bf269b06c537f77ea277041fe152a` |
| `src/agentic_rl/retriever/protocol.py` | 1982 | `cc40c8e887f4e301c2937fbec6176e1db135c18ca3a8166009ebc7f1568c7705` |
| `src/agentic_rl/rollout/__init__.py` | 320 | `93f69bb07acb4304152798e0a6555ef46c2d8cafb9a058c0a70574dba3a0b8b8` |
| `src/agentic_rl/rollout/agent_loop.py` | 3093 | `6eea80d1f931d07159e7d24d83848b6561de4041a1458ade4a7b7bcbf0e06631` |
| `src/agentic_rl/rollout/search_role_provenance.py` | 19574 | `ce811c665b914a57ce308e16ad311681c7c35daa13c1456329e56dbdd497ef7b` |
| `src/agentic_rl/rollout/token_provenance.py` | 3381 | `7b253a5881c9a0beb1343598d5e94f4a5bdedf63c2679e3993414f4bddb826b1` |
| `src/agentic_rl/rollout/trajectory_schema.py` | 25751 | `885677df7d0ac2e7b0341594f23b9baccb55a12204425f6e0ccd745ff61e34c3` |
| `src/agentic_rl/rollout/vllm_manager.py` | 2624 | `497a365e83eae4ffb48319db313da983ccd7c1e72d918238c6927f001f230ee9` |
| `src/agentic_rl/runtime/__init__.py` | 725 | `831726e72cf248b236926bddd60917d4302927f6bf42a06fe1b1fd3be017d7bf` |
| `src/agentic_rl/runtime/async_eval_worker.py` | 19413 | `6862040dd1563aa91daa6b148450adfa8194166fd32c045f93436aa93fc769e0` |
| `src/agentic_rl/runtime/capped_vllm.py` | 29560 | `29da4d78680a80e75629c549ce1d075c546f84aac74ad4500de020b5a56856ee` |
| `src/agentic_rl/runtime/entrypoint.py` | 491 | `b89c28df1a5f9ac848cb3b77487efe614d680b1f908a5ffc6d0bc4b0347bb68d` |
| `src/agentic_rl/runtime/fixed_eval.py` | 6321 | `bdae94ea5243746f826c75930637eb01353df75a127ddb281f5886a6f95b308c` |
| `src/agentic_rl/runtime/formal_state.py` | 6516 | `0b3ed21fff94fa9274074eb3097ebfe80116213b518d049533d32728f7368c37` |
| `src/agentic_rl/runtime/fsdp_worker.py` | 121900 | `8eef5c08809b080c0c727b6fef8876f5e11f87dc2b5043d24ebe110558595b7d` |
| `src/agentic_rl/runtime/learner_batch.py` | 22565 | `6f03c822e9c9f6253cebe03d44131aca014c4304be39102a9f29456da4cb333a` |
| `src/agentic_rl/runtime/postprocess.py` | 11336 | `d7eaece236a674e227290fe93967968ab041fa532d64543bf46f4e8338d888b8` |
| `src/agentic_rl/runtime/pretrain_controls.py` | 3292 | `4e61e6cf5478035e32891b57304350ec8cb16ba54f1dec20bfb9bca33e56c17b` |
| `src/agentic_rl/runtime/ray_topology.py` | 16895 | `34489ca350bfda3d8cc7efb497776939737f1872ee9a2405e97793f8fc43f50c` |
| `src/agentic_rl/runtime/resource_guard.py` | 15283 | `f8a9e1f39795d2e4b51c00da49812fce891a328bb089879bb237a22371703808` |
| `src/agentic_rl/runtime/search_agent_loop.py` | 31492 | `084008de78ed0a8da53d22a5d6a83178c2ff1088778ac444032d902f0a954a0f` |
| `src/agentic_rl/runtime/stop_branching.py` | 57123 | `dbc79ac154d6882b2328001a4f02d50182f56ca1f676776712331b0bc889591b` |
| `src/agentic_rl/runtime/verl_config.py` | 14639 | `ec033d9f246ccbee669b1099f6ac642bdc1fa51d8813a9c16508087389c17ebd` |
| `src/agentic_rl/runtime/verl_runtime_adapter.py` | 310157 | `fa1f09cbb32950712c2cceade321d38bda7ef773bf191af37a42b476940a38cc` |
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
| `src/agentic_rl/workers/resource_plan.py` | 1670 | `76f1aa8a58d2ee2acaa0bdd52e3454a91296a1b333e5e19d88f49d8be129d4a8` |

### tests

| path | bytes | sha256 |
|---|---:|---|
| `tests/README.md` | 803 | `1605014e7f7e546c75d83924ef45bea72f50d979562f21f4a9ac3c308c853329` |
| `tests/config_support.py` | 882 | `c0260abe9c5e2c925b3a42950460c9207f91f643dd656b867575023d6d323aca` |
| `tests/fixtures/base_5x48gb.yaml` | 1180 | `1c4bd3ac03dfaa434bab9f677e00bca4ec0a8b71b57c27b8d1f50ed1906f0648` |
| `tests/fixtures/formal_resume_u20_3rank_4x48gb.yaml` | 174 | `067b9e00e760a9b6a3935ff5a6623da7ca8e4b5c41aa245e5f88dceb0b108f64` |
| `tests/fixtures/formal_resume_u20_to_u500_5x48gb.yaml` | 262 | `d86babc55847a53a3a72eb8275ade30b623490e241f679c8b52a03f151ad04d5` |
| `tests/fixtures/formal_train_5x48gb.yaml` | 303 | `8ad9660b8f28c90cc5fff5c7c1193c94e4a400d75e5e39ac46ca5aa6a6ede813` |
| `tests/fixtures/formal_train_5x48gb_base.yaml` | 154 | `e7926c4c3bf89c82db115a26ce2a78b498059527f930a9cd6714ae9ca9e17266` |
| `tests/fixtures/formal_train_mica_4x48gb.yaml` | 248 | `84b8e5ba5cf4e1cca4519bd1691cffe37d9744b8118b9a4a7f8e9fb1631cf4a6` |
| `tests/fixtures/formal_train_paper_mica_4x48gb.yaml` | 195 | `4dcfdb612919b0919ce0d47b061da0a802afa85f4e30583bee6ac8b8731021fe` |
| `tests/fixtures/pilot_20_5x48gb.yaml` | 236 | `9612e025af31616e6a6f54e8fa54d63b7621a006326e3c9ba5400cd4e224ffa0` |
| `tests/fixtures/reference_4x48gb_resolved.yaml` | 17210 | `45f7cbd1d7b92a8a3d3540a79fa5b1edbb5c02f1e15be3416baef41064e0b52e` |
| `tests/gpu_test_guard.py` | 381 | `89e7b1d626f48868ef99e49382835c797cb2ddc3125b8daa3b7e2fa4ea85d700` |
| `tests/support/exact_ig_fast_path_audit.py` | 81143 | `0f44ddd15b035487a594715bf51673bda3753546baec974321ea11c51e894519` |
| `tests/test_48cpu_resource_profile.py` | 2770 | `8d0a14001ecd72fbc8842c713e929d8ff872282d6eedee6ef371d42376f52fc4` |
| `tests/test_a2tgpo_advantage.py` | 9401 | `67a2451d0c70e53ebf75c740506ace0649f96dca19ff5c266991a50347ec149f` |
| `tests/test_answer_only_ragen2_mica_integration.py` | 16416 | `d0bee5fa1b0a01a6fda21c216e2202f18d2cd48d1faba45fedfca61e8bd81dcb` |
| `tests/test_async_retriever_client.py` | 2677 | `47c74064b1de6efae6efeecfb242cd7fecf0353062f9dcf91c54e134d5b548c0` |
| `tests/test_config_schema.py` | 8095 | `bc07855bf0e8298bdd7a68b139e015e700d01737d1204d6485e6fbfa6aeff359` |
| `tests/test_dataset_view.py` | 1600 | `66a57c987800e61425d5eb30452b0baaa118f42a7925b55eeb0430d434051d54` |
| `tests/test_exact_ig_fast_path_independent_contract.py` | 9699 | `1da5f0ab5455644978be2a6c34f75c02de770746c4abdd75b1147f12675c2873` |
| `tests/test_exact_ig_precision.py` | 7749 | `20532ce55af4cb40859bef1c5b17da735f158a7cbfc9f0f015ce7d8241051269` |
| `tests/test_exact_ig_structure.py` | 17018 | `476e8775f9da17163110e337bbaad22900773d117e3adbf8972c1be4b5f3609e` |
| `tests/test_exact_ig_v4_fp32.py` | 10101 | `031a22082c511526f83a9a3e23a8c4963d6c3c8f23906b37f674981b3301cff6` |
| `tests/test_final_asearch_production_contract.py` | 9241 | `9c4cffb34e550e89542d87789cd81f2eac19d567e4cd7b0afe6a6b70844d2fd2` |
| `tests/test_final_pretrain_controls.py` | 3716 | `1ce7002fb5cc07eb91e3c3ccb1f38dd45b9ab5cf85f82304c7e8e330ce465414` |
| `tests/test_fixed_eval_full_manifest.py` | 2694 | `d672b557a60dfc4c890942cff402f3d7c5f1387b77a3d827937c83d04402e6cd` |
| `tests/test_formal_resume_runtime.py` | 13271 | `764c918c9175ca408a91fe0c80b6c1ef1f0c941b3770b93d5a76334e61b0d798` |
| `tests/test_format_advantage.py` | 4810 | `efaff86d140521e142b2c003022d9ca7c16940692099f1415ae496fb00822c06` |
| `tests/test_fresh_formal_sc_launch.py` | 2090 | `b8797e73ed65618c56d728e550ed35871f20d18aef57078530ae7dd42e8e100d` |
| `tests/test_igpo_f1_parity.py` | 1651 | `d2abb6cd8637921af94e1529d8dcf3c9607abb4d6777c0453a17c5d46f53f4fa` |
| `tests/test_mica_ig_v1.py` | 10622 | `66a9f40d6b553fa438955808a5ec409a4d60c09f338733d537cb1f3aec3f37ce` |
| `tests/test_paper_ragen2_selection.py` | 8900 | `e37304b2e2663743d45b91f68dff0b506211a2acc918b0f2ca78dcd7d1567c71` |
| `tests/test_policy_reduction_and_kl.py` | 7385 | `3c4b082bbfa588d6038a6016874d156cd8f8773bc550c141f1b4b643822c00c6` |
| `tests/test_prompt_sampler.py` | 450 | `06abb7b2a7f0d1d9e0050a319fb4e59f64dff2d6283c1a99f58b8511e53a3341` |
| `tests/test_public_config.py` | 828 | `69ca8c8dee34a3dbd9e3fa98ed5fb747c6a8f2a7b91ec565af36760b7fa7e381` |
| `tests/test_resource_guard.py` | 4689 | `adfe8461fa6c3c847056650130775713127399db912fcbf28cbb61ebbace502c` |
| `tests/test_role_localized_gate.py` | 32392 | `1be5b1aa5a247bff44348bf7735c2b5bc4646abc4af49291466b81837d106e28` |
| `tests/test_runtime_adapter_static.py` | 15824 | `4dba92615f0c955649c9f96df88b43246b9b35dcc0386c5a0166b9f60b4061ed` |
| `tests/test_runtime_learner_batch.py` | 2029 | `bd5294eb8ec58176458d1e92a5c1f8ae59fc86b0f641e0b51fd7f9b447f101e6` |
| `tests/test_runtime_preflight.py` | 548 | `83310072bbc1bea7a44e49ad6bb26fceacae1674b8de2e9c9a72ae7f2de71050` |
| `tests/test_scale_update_stages.py` | 7459 | `c62032232800cdc6c6ec98b7b9c21ebb93a147facc73e5d3d1e629fb37ec8da6` |
| `tests/test_selection_boundaries.py` | 3891 | `dcb74be9d0c099e3d0591d0b36cc9958dffd4a81b8df278da46224a9e1289098` |
| `tests/test_selection_math.py` | 6788 | `6d0423fe27153d11388c746bb19b04bb121b8016d38e8223796163ffa53bd1b0` |
| `tests/test_snapshot_checkpoint_contract.py` | 3789 | `c9cd0a265ef1abc6e7b56fbecf10533f3f416babedad426f2806d55861bc2143` |
| `tests/test_stop_branching.py` | 22763 | `6aeb327d4ba8e543f3043b851fefd752c7d48204cde357dbd88bbc16dcda8745` |
| `tests/test_stop_continue_advantage.py` | 13167 | `d01f9bc5e075dc558c4aeeaee6adec516ec438d652d4e5614f68c0627c8f071e` |
| `tests/test_strict_one_step_contract.py` | 5673 | `da22561fa55041a9bb1a864254811740c3ff1309a8110bc4824e43a447c60111` |
| `tests/test_sufficiency_novelty_cumulative_probe_routed.py` | 16944 | `59e8ac0f37279d2b23ace5d77ca776def6a6db016a6fef074bbb831143e226f4` |
| `tests/test_sufficiency_novelty_local_ig.py` | 9272 | `e7ffeabeae8c69cf7d9a9e7bb04218789ba634b568e131360d58b6736282ce31` |
| `tests/test_token_provenance.py` | 6861 | `cb50c8b20251305b4de64ce3972718ea0bb0544c81fe7f18ad5ded0cbf6d5421` |
| `tests/test_topology_decoupling.py` | 15668 | `ecf06f6e23510580b663437111323265fef32e1b7fb5149db469b1ba117dc0e3` |
| `tests/test_update_controller.py` | 8296 | `08a9ea06361ee35d50dfa82c1b69db7c12aefd9035e27ede0529f832ee820bcc` |

### third_party

| path | bytes | sha256 |
|---|---:|---|
| `third_party/README.md` | 776 | `ceddc84af12f44ef4df96c188b9c9403b7f9b6a2e59d435cf8138a61940dd9a1` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/LICENSE` | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/generation.py` | 54797 | `7019243992a3b70fe4d74d3fff6808cb3d7e25fa0b9ec461a4effa41681333b0` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/prealigned_vectorized.py` | 23073 | `636edca70e84408a988bfb2ff6c7ea0747d3f2bdbc5591d0ad34e3813c963cc9` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/vectorized_gt_logprob.py` | 36023 | `a00da4b594238baa9b2fef911fb5d0a418c5c258d5097559fff3fc6389689f9d` |
