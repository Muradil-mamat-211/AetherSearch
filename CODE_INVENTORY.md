# Code Inventory

This inventory covers every non-ignored file in the AetherSearch GitHub
release. Model weights, optimizer checkpoints, eval result bundles, report
archives, and runtime snapshots are not included.

Generated UTC: `2026-08-19T07:25:43Z`

## Summary

- `configs`: `32` files, `163845` bytes
- `documentation`: `4` files, `11378` bytes
- `environment`: `5` files, `11465` bytes
- `recipes`: `2` files, `3170` bytes
- `references`: `3` files, `55130` bytes
- `repo_root`: `3` files, `1441` bytes
- `runtime_assets`: `2` files, `28709` bytes
- `scripts`: `91` files, `694865` bytes
- `sft_data`: `7` files, `17029` bytes
- `src`: `84` files, `1248902` bytes
- `tests`: `37` files, `308514` bytes
- `third_party`: `5` files, `125893` bytes

## Files

### configs

| path | bytes | sha256 |
|---|---:|---|
| `configs/README.md` | 691 | `0f4b17e623efc8058c5d8980dc7b6e7e9be1899f84d34ec463c4ef82fbec82ee` |
| `configs/base.yaml` | 7215 | `47b0a8cf16332e3a49fe2fc7b9471e7c050cc695d4dfa26e07877d272ac9900a` |
| `configs/exact_ig.yaml` | 4275 | `949ca3d5ada0d442bb89bf91ad47aece7dacd6d46f3e514febf130c21a6951ba` |
| `configs/forced_refill_96_test.yaml` | 84 | `41da07ebaf0143c99c123cc4f63dcba01fbbda5d1969402c6b0ab985f4b97427` |
| `configs/formal_resolved/formal_resume_u040_to_u500_answer_ragen2_mica_ig_v1_g16_20260812_030537/launch_command.sh` | 654 | `5bcc26e468bf7c7b1093fd32fafe46686c821f641123d65d33ae51f404331701` |
| `configs/formal_resolved/formal_resume_u040_to_u500_answer_ragen2_mica_ig_v1_g16_20260812_030537/resolved_config.yaml` | 17800 | `3d3e1da7428dd48195aa41210784dcdc445e7e649d9fd3b6c92f4c65dfbdecff` |
| `configs/formal_resolved/formal_resume_u040_to_u500_answer_ragen2_mica_ig_v1_g16_20260812_030537/source_resolved_config.yaml` | 17729 | `416a319a19471eaea1049da44dd760551ed0faaec5ce4f8257e029e9b80fb37d` |
| `configs/formal_resolved/formal_resume_u180_to_u500_answer_ragen2_mica_ig_v1_g16_20260813_004642/launch_command.sh` | 663 | `a26f04741d2c62df27cb4a26d177d104cef988a9c6481e43ae360a84c6bb6ac4` |
| `configs/formal_resolved/formal_resume_u180_to_u500_answer_ragen2_mica_ig_v1_g16_20260813_004642/resolved_config.yaml` | 17801 | `e48864b90c27d03a755b24f479c7f6817b7ec43e9d36c996c3a88539f43db4c1` |
| `configs/formal_resolved/formal_resume_u180_to_u500_answer_ragen2_mica_ig_v1_g16_20260813_004642/source_resolved_config.yaml` | 17801 | `367adf7875a5563178d89cffe906009ed82ea2b501146c9c3b69ac760b6b8a53` |
| `configs/formal_resolved/formal_resume_u320_to_u500_answer_ragen2_mica_ig_v1_g16_20260814_183801/launch_command.sh` | 663 | `8b3c6da4ea0e934a47c8ebad4577d292425a54d08f5fa39335b73b2356bec7a5` |
| `configs/formal_resolved/formal_resume_u320_to_u500_answer_ragen2_mica_ig_v1_g16_20260814_183801/resolved_config.yaml` | 18036 | `8336273d4f9529502198f687a4d5b9e193d6d289ce37452e7d73ce5ff8586491` |
| `configs/formal_resolved/formal_resume_u320_to_u500_answer_ragen2_mica_ig_v1_g16_20260814_183801/source_resolved_config.yaml` | 17801 | `74e1e7816e1adb0f35f25b960f19d3cc00d3883cb6337a032b7f54be2fd55b51` |
| `configs/formal_resolved/formal_u000_answer_ragen2_paper_mica_ig_v1_g16_20260811_130634/formal_train.yaml` | 233 | `96ce72b40d0bccbb6a8e5683d55db8fd78ddf0500958c88876e0f7cd17dca925` |
| `configs/formal_resolved/formal_u000_answer_ragen2_paper_mica_ig_v1_g16_20260811_130634/launch_command.sh` | 451 | `c16153a8540bebb31af495bd8a0858f182334462c2bfe964e217fcdc80a960af` |
| `configs/formal_resolved/formal_u000_answer_ragen2_paper_mica_ig_v1_g16_20260811_130634/resolved_config.yaml` | 17724 | `a25e21a79129add15938e1365804343b966220cf4c58e3adda236bab14d13681` |
| `configs/formal_resume_u20_3rank.yaml` | 758 | `59aacb416b972e0b0e4925caf28cbde88b8499173e4a277be566d8789aa54d27` |
| `configs/formal_resume_u20_3rank_48cpu.yaml` | 1340 | `5097c3c8f1100531874dfa36a3629212fc50213a9ad2f69fb2bde517216196c4` |
| `configs/formal_resume_u20_to_u500.yaml` | 1774 | `72239de195b48935da947faaa94d3d6aa9665cb97080c36633718209e6b15748` |
| `configs/formal_train.yaml` | 3116 | `bfcda634cebcace153f7fc494f487a5dd84fb94b72c0856ca370d325e352e154` |
| `configs/formal_train_answer_only_ragen2_mica_ig_v1.yaml` | 3146 | `a73a9e2796cf2a6aecd0c107a1b3c8fb981afdbff03044a230d89236565d22a4` |
| `configs/formal_train_answer_only_ragen2_paper_mica_ig_v1.yaml` | 233 | `96ce72b40d0bccbb6a8e5683d55db8fd78ddf0500958c88876e0f7cd17dca925` |
| `configs/formal_train_role_localized_gate.yaml` | 1528 | `5f8e41becdd7d05d2997e0be7622e6a3013ad19074d08d27f1ab4327bbafe552` |
| `configs/gate_calibration_role_localized.yaml` | 1605 | `24c23b9525530fb0ba445f86baecfd565bf7d9ff5108828956ad40a6c79a867b` |
| `configs/hardware/4x48gb_3rl.yaml` | 1023 | `318421ae8a12a8896f7d454f92c741f26ca14fa742452cf14b4dade5abd93a5a` |
| `configs/hardware_5x48gb.yaml` | 699 | `f8f3244b33245d41223f162fb04e958c4486400bebb01252539e626a1cf57b14` |
| `configs/logging.yaml` | 1389 | `59e27ba62355340a12bc0cb5fce372b5cb42f8722aa9d6a339c6b28f8e6d26a5` |
| `configs/pilot_20_final.yaml` | 4399 | `a8d1b07c35d82f9d56a65209bfe7ce81719f467c3da2bfa8ab27fa75659a9f4c` |
| `configs/retriever_external.yaml` | 1080 | `742b087010af3154be6adc70ed44a6a6eb62168c15c883623da4a9f92bae2ffc` |
| `configs/update_stages.yaml` | 1194 | `5a4c9603e68898708d1d4ed82912ee9892ab3c584e8693842437c684f65ad580` |
| `configs/verl_agent_loop.yaml` | 452 | `4215f59b7472b24c573f313f18d907c78aacf59bdbfab8530ad3b37d82cf59e8` |
| `configs/verl_agent_loop_role_localized_gate.yaml` | 488 | `596b89c8e7210a48c2bd8e404e0170fac201ecea4e67d93a08a23d8f96f2c8b3` |

### documentation

| path | bytes | sha256 |
|---|---:|---|
| `EXTERNAL_ASSETS.md` | 1084 | `06f0bf6a6845a7223eb5fa01602dcf92f90e3f3ec33e3240978ad9e8583354d8` |
| `README.md` | 4802 | `ee523fb5b2a3bd4e5157c41cdb982bc2921d5b9a4900987a7e8e8105ad1e4392` |
| `THIRD_PARTY_NOTICES.md` | 4092 | `09c4e8e0e4c97f56361c9bd97959398b21a9ff27f0497e9ca7a6cf0cefc3d605` |
| `TRAINING_REPRODUCTION.md` | 1400 | `ad38126e1bf5265a0a94bf2a17909f0f2ee3c119ce2120b846f0a239ac9d778d` |

### environment

| path | bytes | sha256 |
|---|---:|---|
| `environment/RUNTIME_VERSIONS.md` | 475 | `6aa7fd7a11d47b8d78f5714433e6a2f46e4b78d89c706e8ce641cc8c1de7a0ca` |
| `environment/env.template.sh` | 2393 | `347db23489ad964e192e70250b1e05e5f6507eae2d7192d002a336f734c400da` |
| `environment/requirements-core-observed.txt` | 312 | `9873346cde73b88f492bed5bc7088011b27c71eb137b3a8f5ef402ff14277324` |
| `environment/retriever_pip_freeze.txt` | 3812 | `3d5f5a804cfcc3d31b72b10a81a44a40cd99808282d620106b363b3bb9dc84ad` |
| `environment/rl_pip_freeze.txt` | 4473 | `f1ffe7736d219a7fbe4b57456e0d8200ca8d5eb9f1b185a0d8fb5e4dafbd8c0d` |

### recipes

| path | bytes | sha256 |
|---|---:|---|
| `recipes/rl/README.md` | 1667 | `9208bc27a5252944a2828d1e4e7806292253d6c4c030b4ebbe78cbbe6bc9800d` |
| `recipes/rl/train_4x48gb.yaml` | 1503 | `5f56d5ce132691b2b2e9fdd4dc736706864d2f093f866713975985b01a199e68` |

### references

| path | bytes | sha256 |
|---|---:|---|
| `references/AGENTIC_RL_IGPO_RAGEN2_A2TGPO_SYSTEM_DESIGN_V1_2_UPDATE_STAGED.md` | 51625 | `c32ee11d65cc5a841ab273ba3e5c9587d2edbaed18ba96723445306581107af6` |
| `references/ALGORITHM_OVERRIDES_V2_1.md` | 2398 | `bec2b01855153bb31024f1b76a2d8403438222f4cc9bd0ceec11f46ecc9d8737` |
| `references/exact_ig_fast_path_audit_status.json` | 1107 | `de7b1386e33e9743f930cb398c39e6445055fda1a95cddc761e3d9bc184ff8c8` |

### repo_root

| path | bytes | sha256 |
|---|---:|---|
| `.gitattributes` | 220 | `3a62a151b7887ee92f18a8813b4dd7257ae8864d820910233fce1d6b799b6a88` |
| `.gitignore` | 522 | `7e46b7f10dfc419fde3ed30ab797d192fbf4c806bd20cf1de411f236a44f87f3` |
| `pyproject.toml` | 699 | `9d56ccd76f2c5b6ec4ea56022340c87fe070e3ed9ba02801ed2ec9df48efb9cd` |

### runtime_assets

| path | bytes | sha256 |
|---|---:|---|
| `runtime_assets/retriever/hybrid_retrieval_server.py` | 27862 | `e04a6cc3fe2b90fe58049d703419c1c7759136f438b42cf2b5cb866e8adebb72` |
| `runtime_assets/retriever/retriever.yaml` | 847 | `c778053414b5754ad86dc1ce4e4ad6c96361c3489bd4818a30ab18b8a965c918` |

### scripts

| path | bytes | sha256 |
|---|---:|---|
| `scripts/README.md` | 1105 | `420f7f7c32465de553b5cc826a5cfcc41b24446bb6e70111ac5e4613f72c639f` |
| `scripts/_formal_retriever_process.sh` | 654 | `383c12ca79e5ecff826357832afe9c04587fb7809e18908300ee663182417922` |
| `scripts/_formal_trainer_process.sh` | 2670 | `69da2580b5da64e20f13103b29374fa02e62f429b3f0d5908f76d46293882dcd` |
| `scripts/_run_runtime_job.sh` | 5064 | `87b5eae9ac488eb5cf743977578fd1bb9a304722b2320fd6e8f0c9d3f0d21162` |
| `scripts/async_eval_gpu0_worker.sh` | 869 | `c606d034f9dc66feb1f926ca7b426514da285c644cfe63d49841328ee74f858d` |
| `scripts/audit_asearch_lambda_ig_03.py` | 20218 | `e33684156ea0a9483fc5300db57c37b700eef3ed82d05822233eafa87131eeb3` |
| `scripts/audit_exact_ig_fast_path_production.py` | 81208 | `0f2b18ba111643c0880c2e5adb0464765a9661cca8de4c06ca6f804c92433e38` |
| `scripts/audit_exact_ig_target_tokenization_v3.py` | 7252 | `6e8db2472d91480f5ab06db53b9dd6b333e6cb3585fceefead6ae88c69fd1989` |
| `scripts/audit_final_algorithm_v2_1.py` | 22278 | `49b9406d46471e99a4ceef285d41390beb1ead922771e57c825b2776cd4e6235` |
| `scripts/audit_final_asearch_production.py` | 33611 | `3af15de5fa554cb51aacd3e77cd7f777e408553d441ee6496319844c191fdb37` |
| `scripts/audit_igpo_f1_parity.py` | 3170 | `72ff76522317f5ab09e8f8c5e7b46b91da1278ec546b29072e9dae2fb25e9777` |
| `scripts/audit_mica_ig_v1_offline.py` | 25534 | `cd172c0374d303c0076b2c0e9822a8206854119dfe9933e0183b81c0238a5507` |
| `scripts/audit_paper_ragen2_restart.py` | 7025 | `1819ee72fcf73abaacfcccd942d108bc7f424c827cce53195c0f15ac957190cf` |
| `scripts/audit_probe_routed_search_advantage.py` | 3196 | `139876d7b1eb1729e305a1a0ab32a4285c0cd3fb4a3709310ce4139dd1963831` |
| `scripts/audit_role_localized_gate.py` | 4408 | `c45180c0be0e0a9a220b5f6e32ba6433918675e7651dca84c7eca7a502dd8809` |
| `scripts/audit_role_localized_gate_root_cause_u1_u139.py` | 59542 | `19695aee515c2048b90e55e5805b9b3bd6714a325eee7dfc1b2df62a2bb9ebf3` |
| `scripts/audit_static.py` | 5285 | `84da553654031bac7ebe6bc699db8e72418a2f7244ad59c43aabe4b5cbe5138c` |
| `scripts/audit_stop_continue_search_advantage.py` | 6229 | `abd778072d0cfce2268a8b28f11c0fe77470e6d56f907f6528318532d2d8cac9` |
| `scripts/bootstrap_env.sh` | 947 | `d1e17970e9ed9746625d67d207115e99ecce8281cfdbc905d82ccfaf9bc07376` |
| `scripts/build_code_inventory.py` | 2856 | `75e4320117711f34b5d892c90699e00948825a75e41597e03ddb194ec1e978d8` |
| `scripts/build_manifest.py` | 2068 | `12d2e24ef42d915cae8b40810f1b1fc76355aedd068bf9da2cf83ffe992e0a11` |
| `scripts/check_algorithm_boundary.py` | 5836 | `02757b9d3252ff495d4e751c201cb1e62237086c841c0786507bc8479d24a5a3` |
| `scripts/exact_ig_layer_diagnosis.py` | 168 | `35439412616e05bd05513f94b6eea01e63acd32f7a5e9376f67ae88cf55b4384` |
| `scripts/exact_ig_numerical_parity.py` | 227 | `9147e8bc65c3e3a33e858f10232198c748cae8c741cf68f449b98adc9d5aad07` |
| `scripts/exact_ig_precision_diagnosis.py` | 227 | `9147e8bc65c3e3a33e858f10232198c748cae8c741cf68f449b98adc9d5aad07` |
| `scripts/exact_ig_ragen_equivalence.py` | 199 | `bf341e1efcc22ab366760b479547aefc91fd35d66e3ae8c7369a9b16142c412d` |
| `scripts/export_effective_algorithm_v2_1.py` | 25336 | `93b74d563b2ba14c7855471b75525d9695b08d142453b120f271689a861feaa5` |
| `scripts/export_exact_ig_batch_budget_tests_v3.py` | 10159 | `ab023e348d35591604c57493d2f14e4a2a7c5764f68ebe1fee870622592491f2` |
| `scripts/export_exact_ig_batch_budget_tests_v4.py` | 10252 | `0d61a4483a153f474f872af7c48dc6314e7ce9ffca9e8acfea5545679116198a` |
| `scripts/export_update_metrics_u1_to_current.py` | 9585 | `57a3283020f03d0c6675b21f77ba18233f988b3ade9d2fcdc57681bd564a19a2` |
| `scripts/finalize_final_pretrain.py` | 25952 | `479963eb2bd44b3cc1902d898ad66fcadd9b65e0a51262d7555770daf6437068` |
| `scripts/finalize_preformal_runtime_qualification.py` | 30403 | `ff8bb0c99f9af7192b32d6a844f9a16645bf401e7222a48fc1b0506b8d7d07ff` |
| `scripts/finalize_role_localized_gate_config.py` | 2806 | `a249b879d931117c98998a2832d4c3d1537bfb37f451c825e7b486f66c91f7f5` |
| `scripts/formal_training_watchdog.sh` | 767 | `b523d926e0d1d0a50d6bd9d5026ba5264c503f647f4cb724f06be8be123ee479` |
| `scripts/launch_retriever.sh` | 3411 | `e5b709d0f9ec7f21964795506faea818305d35713ecda71e6bec870a1bca38ab` |
| `scripts/launch_retriever_gpu0.sh` | 159 | `e000c23b1d3fa1f5d010c083f67a2792b3fea6e010614dde9449b25ac6a1e9eb` |
| `scripts/launch_rl_gpu1_4.sh` | 581 | `8e31cfe65f0913e4b11d91964a5cbcefc19766fa186aa5b4c6d3a5279cb6982f` |
| `scripts/launch_train.sh` | 151 | `25d8ced298d391f49158e1af29d8bd23ac08e99e3d78c5434237badfe4571b9e` |
| `scripts/monitor_formal_training_10min.sh` | 857 | `8bae99da164f9351ed5295c989449e598dc0a2b321b5f3126e0a8d759f7e07b8` |
| `scripts/preflight_fresh_formal_sc.py` | 10971 | `d8dbc543a69c1593ffc99620a5578977086acc66eadd22c019212bda643f37c8` |
| `scripts/preflight_mica_formal.py` | 7277 | `4c93e6d07905700076e7e049b93d81133c99d37d362483ad5851adb63cb706d2` |
| `scripts/prepare_formal_resume_from_checkpoint.py` | 8404 | `b7784b402421b5899aa832618044102be20782cefa593437f542e916acff962b` |
| `scripts/prepare_formal_resume_run.py` | 11837 | `d609bce7145e9b28cb577980b0d2e968713e60c2e508f2c594fd06a452121aac` |
| `scripts/prepare_verified_resume_recovery.py` | 13841 | `3a815e65163dfeb8ab6525806720118c5ea26f378e4d12a0dac979211917063c` |
| `scripts/print_resolved_config.sh` | 365 | `6632c076645aa662f45c62f74614cdd89ce45a549d7748dbb02a996b1a616818` |
| `scripts/resolve_mica_formal_config.py` | 1645 | `a0de4c38f746dce5ac10d3ec0ba64efb31b3fa305c8b7c0383d8bee8ef96a5a1` |
| `scripts/resume_formal_from_u80_no_monitor.sh` | 3646 | `d0285759875d7a7c65d8278930add31adba1e81376eb57fbc985f1b0d8c9c383` |
| `scripts/resume_formal_manual.sh` | 2386 | `e6f6168d585e9e3a139107fae451b414138f62b4e427b06e652cfa8217cbd901` |
| `scripts/resume_formal_u20_3rank_48cpu.sh` | 6454 | `727a126b0df435a3f02038bf3eebabaa9644221c86288eb9b41a03ab9afab0cd` |
| `scripts/resume_rl.sh` | 176 | `6bb3b7b6f3f15167222d0aff0bb73efeb45e32dcc04ef21631ea549e145067ab` |
| `scripts/resume_verified_formal_checkpoint.sh` | 8449 | `99dadb67a296e7e33799d238cec1dd8b14745ba4b967797434a012fe16fd61d9` |
| `scripts/run_exact_ig_fast_path_repair_validation.sh` | 259 | `137cd941e46dd4ab79b5febabc2f96840087d261aa7c12d3f90885f0095e9bb0` |
| `scripts/run_exact_ig_fp32_fast_profile.sh` | 153 | `95d3ff5023e407c35f51c6a73b9378bfc035ec5515db6926838556d6ae585f8e` |
| `scripts/run_exact_ig_numerical_parity.sh` | 220 | `ff333941fcbbdc076852ee51e729c1c6d4f0cebcb076f5b8af95a88a9f75930d` |
| `scripts/run_exact_ig_precision_diagnosis.sh` | 220 | `ff333941fcbbdc076852ee51e729c1c6d4f0cebcb076f5b8af95a88a9f75930d` |
| `scripts/run_exact_ig_ragen_equivalence.sh` | 216 | `b87d5f4f947bf62a62ed0e9d9314286c6e3b432044c39b4fa0052ce96c1a8a87` |
| `scripts/run_final_pilot_20.sh` | 5879 | `adbe637a2de78dde40578680386627eab7be0e649be3043acbc9b209a30239f0` |
| `scripts/run_forced_refill_96_test.sh` | 3488 | `51028e279a704bd2ba8d8665e8504276f3ab6251bcf7f3ee982f292a6e3b23fd` |
| `scripts/run_mica_preformal_gate.sh` | 4169 | `3d1d8a366c804fa285d54067246c84f12c5c8094d1380d4c0a15914d81403407` |
| `scripts/run_parallel_smoke.sh` | 2803 | `81279be5be018e7c679047c848da15144f47d2b48ab72b0fdca505092092337c` |
| `scripts/run_pilot_50_updates.sh` | 1215 | `2bf012beaa676ba2289cddae395b7bdfc67b6314ad7ceac0cc55c4d4c5114b11` |
| `scripts/run_role_localized_gate_calibration.sh` | 1781 | `d2dfea5f2184bcc6237a78f570e70d930e0ccba001fb0e6d4b88e816fbb4b688` |
| `scripts/run_runtime_stage.sh` | 3107 | `842e03db630c8ca5f2fd9fee7eb7cdc31adb40bb8df784498c3baf8eca199eaa` |
| `scripts/run_smoke_pipeline.sh` | 1433 | `4a4ab57a10b785d5c10f30486c3a619f5bf5b22673b677b173e0f0cabce061f1` |
| `scripts/run_stage_a_tito.sh` | 158 | `2ce7c52347f443a5365ad4aea9900f42e01ead7e0952de8708265c741cb5fa74` |
| `scripts/run_stage_b_one_update.sh` | 158 | `8c156c9b91de04b33de44f5ea0681585146755e8e6fd77a6ee954e321ed21d53` |
| `scripts/run_stage_c_five_updates.sh` | 158 | `090f00275f8e70a5826483b1c51f4d542a6e5f45f10e3bdf6a33a4ad5262568a` |
| `scripts/run_stage_d_full_shape_one_update.sh` | 158 | `1bfed097310b1138b3b55b1b1b26bff605463353d734eaace12e885f10ee7b44` |
| `scripts/run_stop_continue_no_update_smoke.sh` | 193 | `fd91b0110de3a469c917ae89e6ad8fc121bf39440a6d0b395a2b5261cce0fc68` |
| `scripts/runtime_guard.py` | 5467 | `36ee203cf22ecb33b8560b268f9add0a475981b7dc01117bf8f2c019bf85b529` |
| `scripts/snapshot_igpo_official_v3.py` | 2748 | `9daa32909efa6b1f2c7e99a93da25219fde31ac5358eebfeb767bfc3e4e6e782` |
| `scripts/status_formal_training.sh` | 1900 | `9c6e175dc11a66015617719619f3c765a7af41dcad370cf9d0f91b390698bc22` |
| `scripts/stop_formal_after_checkpoint.py` | 3613 | `5ec8707f46aa2838d91c1d6b1bc904a448ff0aba142a7390fca3ddf1071a0994` |
| `scripts/stop_formal_training.sh` | 1743 | `9c917141d2bd75b2d9078f04bc236b26cd819d207d6ca7fbafbf046afd274733` |
| `scripts/stop_services.sh` | 399 | `62520cfc11a2cd85c42e5f1d1bc97a488d00cf82ad0611f83421914af150467b` |
| `scripts/tail_formal_logs.sh` | 832 | `a616421a0ca253bdb098c371801baf0014452250f75e5f3f1c80e0d9c5cc6247` |
| `scripts/tail_formal_training.sh` | 161 | `c6a7587b25b36ab37f047a532d1860cb3453bf0ec60ab31c8a6603181b8d3076` |
| `scripts/test_code.sh` | 626 | `d29c3c344d51d09b649dc0fcb2321e26f82672478b6c7850bcf2831df2fc4f27` |
| `scripts/train_formal_from_pilot20_to_500.sh` | 11445 | `3f747b7d99ad81e45eae9d74e27e7957fbc866ebdc14ba1d41e84569b23380b0` |
| `scripts/train_formal_manual.sh` | 7868 | `6ef132965cefec843b138f42fd012aeefe759676c53cb8491887ebedda31542b` |
| `scripts/train_rl.sh` | 2206 | `69b94d126110f4c70a53b7726e44cd4ca566737d8d10a09edcf8ace44d1b5fb9` |
| `scripts/validate_48cpu_resource_profile.py` | 4811 | `d02248f655d0ff0d2745fb21d03e089e177365cfe89d2d3678ce834da8df8823` |
| `scripts/validate_exact_ig_fast_path_repair.py` | 931 | `1faa78cdbbe1c43a5b654ac2967086db604b4b7e976e626512c7171c3c8a864b` |
| `scripts/validate_exact_ig_fp32_v4.py` | 51099 | `2ab1936b2007ce8ca865040eb1cf4fdae777f00f75b86836f9c41bbdd571fffb` |
| `scripts/validate_exact_ig_fsdp4_v3.py` | 14997 | `76d94b763694a8370c517b7e51da77ccfdb83954c27586d5f3da17b3d6dfedc1` |
| `scripts/validate_exact_ig_fsdp4_v4.py` | 19300 | `7a687fd5cbcb25e408cb8fd43d42202e6f4aee338eba7447c9e0169e40a406fb` |
| `scripts/validate_exact_ig_official_alignment_v3.py` | 35425 | `68ad98f686663ec2ff22ee3959e487c0d50b3231dc28b309ea9229ac73e62446` |
| `scripts/validate_resume_3rank.sh` | 1197 | `46765340fe500a36ad995f52b7b25acd00b86750506233adf4a4b0256019b160` |
| `scripts/validate_resume_3rank_48cpu.sh` | 1209 | `0574dd2ca060a368d49eb0e185fd6eed5a4516d39e1307fa96fa581e0f61e45c` |
| `scripts/validate_static.sh` | 745 | `cfad6c21c8784ccac8909f295a8908d1099796a6bce87f2e40f4ebbdddedf90a` |
| `scripts/verify_checkpoint_readonly.py` | 2659 | `c8af6683b6f6e09e96fa91c16cf679706b4673d947e5437428239d542e6ad9d2` |

### sft_data

| path | bytes | sha256 |
|---|---:|---|
| `sft_data/ATTRIBUTION.md` | 2776 | `11ca5354884cad2bbc64b224a930d263e741f3d0b9148490152811f9cc24822c` |
| `sft_data/README.md` | 3909 | `1a28778875d73880bcc30130c53eb496fd901327ad1712faac0fd1476cccfa4c` |
| `sft_data/checksums.sha256` | 419 | `e6b1fc982ac115459215e718e031c5b83a66e53bb9c559b7916c7da265b58e11` |
| `sft_data/dataset_manifest.json` | 1444 | `3b6ca2879d1d680d3226ed96c326e1ee1ff566d4eabe73ed005619096e02de8e` |
| `sft_data/scripts/build_audited_queryrewrite.sh` | 4353 | `63d66396282c48cf5a1abe71f53b92fbbb0c54f543096b34dfbe77ca8ffde9ee` |
| `sft_data/scripts/build_final_sft.sh` | 2874 | `41cad559a5d36999db0e1b47de391a8b8a4c6230ccfeea9e0c3e2b07eb642b3d` |
| `sft_data/scripts/standardize_queryrewrite_think.pl` | 1254 | `4f6d97cd6cb28c241f6b5e4807f61f39abdcd628cdaa1d03e661604e65ba2ff6` |

### src

| path | bytes | sha256 |
|---|---:|---|
| `src/agentic_rl/__init__.py` | 78 | `5e9dabeb9b9ec3e1a11cb68b13b73f1f11d514a3dbda94ef7d0a835095371737` |
| `src/agentic_rl/advantage/__init__.py` | 1964 | `2e38f447a0511bc1bbebafa49f994eba2e814bc921aedc55ccf113d5ab23295b` |
| `src/agentic_rl/advantage/a2tgpo.py` | 73550 | `afa94ba2e083b9d1dff4707da3be941b854b2e46fca95769b7ff049774050a98` |
| `src/agentic_rl/advantage/mica_ig.py` | 15595 | `da2c3fe48716ede2355368136b7a995e9c97d50ac8b63ab6aa7bec813a460cb5` |
| `src/agentic_rl/advantage/role_localized_gate.py` | 14335 | `4faa97b10be764a94b118a9948a65918f47b9fa860d2a902180f9f7825f7146b` |
| `src/agentic_rl/advantage/stop_continue.py` | 15470 | `a9d281ce55bde5357ebbd8b1965df85c857cbec07ca4ca4f78f7b66655bf5902` |
| `src/agentic_rl/checkpoint/__init__.py` | 277 | `1de6ba58c69f24283360d5b26175b2c166192462ef8c051c677684cab893e0de` |
| `src/agentic_rl/checkpoint/atomic_commit.py` | 8441 | `68e487e5f7a0ec53f74f26a052d3bfc480755ea38872a06d73d9f8cde582f466` |
| `src/agentic_rl/checkpoint/fsdp2_dcp.py` | 2494 | `48cd9009ab11f3a6a202c3bcd16a9aa128650f577af3a3174cf6a8a0bd4fb5a7` |
| `src/agentic_rl/checkpoint/state_schema.py` | 2307 | `8d0660762daf08cdb1ea14a7077215d86abb1456f335ac2396bb6ea2b47a2303` |
| `src/agentic_rl/config.py` | 45441 | `e4e507e64ee3485d826a67fe20380f967ca826c90a1a2c99a7fbd51866253c2c` |
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
| `src/agentic_rl/retriever/__init__.py` | 392 | `ce13413e3f102a5c67e7f1dd669b8d3d5456af36993a6315782793c249ea7412` |
| `src/agentic_rl/retriever/client.py` | 10941 | `ce0f8ffd010ae3a2f4893b00305296802511506a812ac0f544ddf3f642122d1f` |
| `src/agentic_rl/retriever/health.py` | 1635 | `6ee579cbd85a513f82a85fbd8dfe2423a65bf269b06c537f77ea277041fe152a` |
| `src/agentic_rl/retriever/protocol.py` | 1982 | `cc40c8e887f4e301c2937fbec6176e1db135c18ca3a8166009ebc7f1568c7705` |
| `src/agentic_rl/rollout/__init__.py` | 320 | `93f69bb07acb4304152798e0a6555ef46c2d8cafb9a058c0a70574dba3a0b8b8` |
| `src/agentic_rl/rollout/agent_loop.py` | 3093 | `6eea80d1f931d07159e7d24d83848b6561de4041a1458ade4a7b7bcbf0e06631` |
| `src/agentic_rl/rollout/search_role_provenance.py` | 19574 | `ce811c665b914a57ce308e16ad311681c7c35daa13c1456329e56dbdd497ef7b` |
| `src/agentic_rl/rollout/token_provenance.py` | 3381 | `7b253a5881c9a0beb1343598d5e94f4a5bdedf63c2679e3993414f4bddb826b1` |
| `src/agentic_rl/rollout/trajectory_schema.py` | 25751 | `885677df7d0ac2e7b0341594f23b9baccb55a12204425f6e0ccd745ff61e34c3` |
| `src/agentic_rl/rollout/vllm_manager.py` | 2624 | `497a365e83eae4ffb48319db313da983ccd7c1e72d918238c6927f001f230ee9` |
| `src/agentic_rl/runtime/__init__.py` | 725 | `831726e72cf248b236926bddd60917d4302927f6bf42a06fe1b1fd3be017d7bf` |
| `src/agentic_rl/runtime/async_eval_worker.py` | 19079 | `b5ef930bf12d0eedbc4c92111ba3a08eab30a9b634ff512adf5ec04fe09876bb` |
| `src/agentic_rl/runtime/capped_vllm.py` | 29560 | `29da4d78680a80e75629c549ce1d075c546f84aac74ad4500de020b5a56856ee` |
| `src/agentic_rl/runtime/entrypoint.py` | 491 | `b89c28df1a5f9ac848cb3b77487efe614d680b1f908a5ffc6d0bc4b0347bb68d` |
| `src/agentic_rl/runtime/fixed_eval.py` | 4919 | `fa1b8ff6591e1c47d0dc6ebb38ab4e2694378beb6362d74e2930973c280dd9e1` |
| `src/agentic_rl/runtime/formal_monitor.py` | 24244 | `b882461908033497e172f154b1724a5dbdeb2375dc2a64618bf78b7aaf119947` |
| `src/agentic_rl/runtime/formal_state.py` | 6516 | `0b3ed21fff94fa9274074eb3097ebfe80116213b518d049533d32728f7368c37` |
| `src/agentic_rl/runtime/formal_watchdog.py` | 9164 | `f052ef2332c99fcfaf2727a76022638ea163ee797c28c2f49e220718d549921c` |
| `src/agentic_rl/runtime/fsdp_worker.py` | 121900 | `8eef5c08809b080c0c727b6fef8876f5e11f87dc2b5043d24ebe110558595b7d` |
| `src/agentic_rl/runtime/learner_batch.py` | 22565 | `6f03c822e9c9f6253cebe03d44131aca014c4304be39102a9f29456da4cb333a` |
| `src/agentic_rl/runtime/postprocess.py` | 11336 | `d7eaece236a674e227290fe93967968ab041fa532d64543bf46f4e8338d888b8` |
| `src/agentic_rl/runtime/pretrain_controls.py` | 3292 | `4e61e6cf5478035e32891b57304350ec8cb16ba54f1dec20bfb9bca33e56c17b` |
| `src/agentic_rl/runtime/ray_topology.py` | 15564 | `5d9685f88cc416a30be51bdc71c66532a293c42e20ca7782f507781a99adab94` |
| `src/agentic_rl/runtime/resource_guard.py` | 10859 | `8ea368044ca0144d58638b65201d2b03405610b0d0a5dc301079c2d168fc570d` |
| `src/agentic_rl/runtime/search_agent_loop.py` | 31492 | `084008de78ed0a8da53d22a5d6a83178c2ff1088778ac444032d902f0a954a0f` |
| `src/agentic_rl/runtime/stop_branching.py` | 57123 | `dbc79ac154d6882b2328001a4f02d50182f56ca1f676776712331b0bc889591b` |
| `src/agentic_rl/runtime/verl_config.py` | 14501 | `8680940c59198c9601190ca77a63a80748619ac6546a8dee45800b87be43a463` |
| `src/agentic_rl/runtime/verl_runtime_adapter.py` | 309933 | `b0d3a21da9dbd67356dfe5dc69c95d11340601a0a1a07ee9208e7681050cc916` |
| `src/agentic_rl/selection/__init__.py` | 931 | `bdb9c618bd263a4ffc39a2da3f8f176ccb422c0319731cbd375caafed94165c9` |
| `src/agentic_rl/selection/candidate_pool.py` | 15734 | `c88b723b6c14fa538989017d613961231ef9d9d75d214fd6674211c3b2aab192` |
| `src/agentic_rl/selection/channel_scale.py` | 6344 | `b869bc5b269be447465e9c8a12532593bdfee45280b77d5a091dc58d450a2991` |
| `src/agentic_rl/selection/health_gate.py` | 1158 | `768ea639bee3163340a4730a913de61a343971e57912e471b33d414cba1a7649` |
| `src/agentic_rl/selection/paper_ragen2.py` | 1235 | `b61a658f62d3da244078f68b0af61df9a6fee2838ccd8450c4ca7bcdfd5e5b1e` |
| `src/agentic_rl/selection/prompt_variance.py` | 4169 | `e0709b1e969af3ee72478e1234ef886f4099f856a4a917da246c94b057cb9b2d` |
| `src/agentic_rl/selection/top_p.py` | 1782 | `65cf55c3d9c616ed87fa8f64a62d0c3cae67d0e005d806aed2127ad96ca8a6b6` |
| `src/agentic_rl/workers/__init__.py` | 174 | `adda67db60b379cb6126fd0620076545ae5f715a5c3011b08c32054a5a28536c` |
| `src/agentic_rl/workers/failure_coordinator.py` | 1105 | `df7bfd662936632e54c4576866c2902ec3a6f1f2623922315f77d02aff5c03f6` |
| `src/agentic_rl/workers/fsdp2_engine.py` | 1555 | `8761971537999008678da2c130241403fd0357451e0d7ca8d520becb3b4a3107` |
| `src/agentic_rl/workers/ray_actors.py` | 15299 | `c7991b5d9df503199cedd4a3a12cbc142a5ed89da79d3aefd6c6bac6ec36f97a` |
| `src/agentic_rl/workers/resource_plan.py` | 2286 | `071b7689d202587309afb66dc5d85051937d79496bb5d6931e9f8492248a8b12` |

### tests

| path | bytes | sha256 |
|---|---:|---|
| `tests/test_48cpu_resource_profile.py` | 2738 | `f62e4e67f33944d20912596880db6b212d21b766ef2cd0e8d277bef3221aa098` |
| `tests/test_a2tgpo_advantage.py` | 9401 | `67a2451d0c70e53ebf75c740506ace0649f96dca19ff5c266991a50347ec149f` |
| `tests/test_answer_only_ragen2_mica_integration.py` | 17130 | `0dddcef78eb8fc06fa8db3ee96efb19c32e9dc97533f8a76a905c957d6cbca37` |
| `tests/test_async_retriever_client.py` | 2677 | `47c74064b1de6efae6efeecfb242cd7fecf0353062f9dcf91c54e134d5b548c0` |
| `tests/test_config_schema.py` | 7978 | `476f2b4935486d285020d64d1d81753b6a06bad499ce1273dee69833eeab0847` |
| `tests/test_dataset_view.py` | 1402 | `0a807f7094014e5ff94942eef6ba1f58637b798a2051a8e9e53dc854eb14cb15` |
| `tests/test_exact_ig_fast_path_independent_contract.py` | 9696 | `ba2d1720a5e91a169fc780b17b990fbcf6962a731a4264a268b30b63f9f4fa2a` |
| `tests/test_exact_ig_precision.py` | 7623 | `7d5bacb486d60a4b7d42515615ea7c0259913a6028bd87e5849f6bd3c5eae780` |
| `tests/test_exact_ig_structure.py` | 16994 | `6eadc30ba1f4ae4677abd871be2d2b40a17efff6fa06c89310aa1e1c0ed352bc` |
| `tests/test_exact_ig_v4_fp32.py` | 10101 | `031a22082c511526f83a9a3e23a8c4963d6c3c8f23906b37f674981b3301cff6` |
| `tests/test_final_asearch_production_contract.py` | 9220 | `86c732d393bfb88e64187e6ef543973d5745fb972cee8bef0a6831a5eca90e2a` |
| `tests/test_final_pretrain_controls.py` | 3686 | `d3626e0e165caf02820c199711c75f9c89730cd71c072a4489227d8f53419228` |
| `tests/test_formal_resume_runtime.py` | 13299 | `85c9e3391aae5182de79a9269668af55e1e3425fa3103d065ae1702b2a438175` |
| `tests/test_format_advantage.py` | 4810 | `efaff86d140521e142b2c003022d9ca7c16940692099f1415ae496fb00822c06` |
| `tests/test_fresh_formal_sc_launch.py` | 2280 | `6b9ab0b29349371ae955af1313fc0b516d0c77eae6a400ac35bce8300f3e50a8` |
| `tests/test_igpo_f1_parity.py` | 1651 | `d2abb6cd8637921af94e1529d8dcf3c9607abb4d6777c0453a17c5d46f53f4fa` |
| `tests/test_mica_ig_v1.py` | 10622 | `66a9f40d6b553fa438955808a5ec409a4d60c09f338733d537cb1f3aec3f37ce` |
| `tests/test_paper_ragen2_selection.py` | 8483 | `ba929267ec2545abd659c1798b4ea6cb1c25b03648715e2f4f75f9550e545f1e` |
| `tests/test_policy_reduction_and_kl.py` | 7385 | `3c4b082bbfa588d6038a6016874d156cd8f8773bc550c141f1b4b643822c00c6` |
| `tests/test_prompt_sampler.py` | 450 | `06abb7b2a7f0d1d9e0050a319fb4e59f64dff2d6283c1a99f58b8511e53a3341` |
| `tests/test_public_config.py` | 828 | `69ca8c8dee34a3dbd9e3fa98ed5fb747c6a8f2a7b91ec565af36760b7fa7e381` |
| `tests/test_resource_guard.py` | 4530 | `271c5a21362296f5985e03ad4e03c5b5a0d37dbdc3d10167c31304129c5a22e1` |
| `tests/test_role_localized_gate.py` | 32392 | `1be5b1aa5a247bff44348bf7735c2b5bc4646abc4af49291466b81837d106e28` |
| `tests/test_runtime_adapter_static.py` | 15867 | `380aa4c6fac3843966a1c341f7c12caa91fa4c1a7843442b502742ef1bcb54bf` |
| `tests/test_runtime_learner_batch.py` | 2029 | `bd5294eb8ec58176458d1e92a5c1f8ae59fc86b0f641e0b51fd7f9b447f101e6` |
| `tests/test_runtime_preflight.py` | 548 | `83310072bbc1bea7a44e49ad6bb26fceacae1674b8de2e9c9a72ae7f2de71050` |
| `tests/test_scale_update_stages.py` | 7459 | `c62032232800cdc6c6ec98b7b9c21ebb93a147facc73e5d3d1e629fb37ec8da6` |
| `tests/test_selection_boundaries.py` | 3891 | `dcb74be9d0c099e3d0591d0b36cc9958dffd4a81b8df278da46224a9e1289098` |
| `tests/test_selection_math.py` | 6788 | `6d0423fe27153d11388c746bb19b04bb121b8016d38e8223796163ffa53bd1b0` |
| `tests/test_snapshot_checkpoint_contract.py` | 3789 | `c9cd0a265ef1abc6e7b56fbecf10533f3f416babedad426f2806d55861bc2143` |
| `tests/test_stop_branching.py` | 22702 | `d905a9bf46e6176740a65ee446a96e061f11d90df08e3471658d373b5b214034` |
| `tests/test_stop_continue_advantage.py` | 13167 | `d01f9bc5e075dc558c4aeeaee6adec516ec438d652d4e5614f68c0627c8f071e` |
| `tests/test_strict_one_step_contract.py` | 5673 | `da22561fa55041a9bb1a864254811740c3ff1309a8110bc4824e43a447c60111` |
| `tests/test_sufficiency_novelty_cumulative_probe_routed.py` | 16866 | `21b56ae70c33598c41237b935373c5ce5d433ad42ae956bf64b693f4bfbcf3dd` |
| `tests/test_sufficiency_novelty_local_ig.py` | 9211 | `b1c5ad8ca5e3a25a7bcba760623e1ab45a44ea833edfb996de727379db20d9e0` |
| `tests/test_token_provenance.py` | 6861 | `cb50c8b20251305b4de64ce3972718ea0bb0544c81fe7f18ad5ded0cbf6d5421` |
| `tests/test_update_controller.py` | 8287 | `306f2f7b65687622826d882a9d777717d2819af34796222632cfd1331ead219c` |

### third_party

| path | bytes | sha256 |
|---|---:|---|
| `third_party/README.md` | 643 | `497e006a537d6d6e2f8b980523cf014bde78389c268b0203f7ee752d73654a75` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/LICENSE` | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/generation.py` | 54797 | `7019243992a3b70fe4d74d3fff6808cb3d7e25fa0b9ec461a4effa41681333b0` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/prealigned_vectorized.py` | 23073 | `636edca70e84408a988bfb2ff6c7ea0747d3f2bdbc5591d0ad34e3813c963cc9` |
| `third_party/igpo_official_64165e2741ed8801f977948c8128080ce87b4101/scrl/llm_agent/vectorized_gt_logprob.py` | 36023 | `a00da4b594238baa9b2fef911fb5d0a418c5c258d5097559fff3fc6389689f9d` |
