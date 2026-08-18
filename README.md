# Clean Search SFT Final Dataset

这是当前 Search-SFT 项目的最终清洗版训练数据。它把 QueryRewrite 和 V3.1 两类数据统一成同一种 `prompt_text + target_text` 格式，便于使用同一个 SFT 训练器训练。

## 文件说明

- `clean_sft_final.jsonl`：最终合并后的 SFT 数据，应作为训练主文件。
- `clean_queryrewrite.jsonl`：清洗并统一格式后的 QueryRewrite 子集。
- `clean_v31.jsonl`：清洗并统一格式后的 V3.1 子集。
- `manifest.json`：机器可读的来源、去重统计和校验结果。

## 最终统计

| 项目 | 数量 |
|---|---:|
| QueryRewrite 原始问题 | 10,000 |
| QueryRewrite 与 Search-R1 重合并删除的问题 | 5,784 |
| 保留的 QueryRewrite 原始问题 | 4,216 |
| QueryRewrite 标准化 SFT 记录 | 8,432 |
| V3.1 保留的问题 | 975 |
| V3.1 标准化 SFT 记录 | 1,918 |
| 最终 SFT 记录 | 10,350 |
| 最终唯一原始问题 | 5,191 |
| QueryRewrite/V3.1 原始问题交集 | **0** |
| 最终数据/Search-R1 原始问题交集 | **0** |

最终记录类型如下：

- `search_retention`：5,159 条
- `final_answer`：4,216 条
- `final_answer_repair`：975 条

QueryRewrite 与 V3.1 的问题集合是互不相交的；同一个问题内部出现两条不同 SFT 记录是有意的，因为它们对应不同的训练目标或不同的上下文前缀，不属于两个数据集之间的重复。

## 去重规则

去重依据是原始 `question`，不是 QueryRewrite 生成的搜索 query，也不是 answer。问题先经过以下归一化：

1. Unicode NFKC 规范化；
2. 大小写折叠；
3. 合并连续空白；
4. 删除末尾连续的问号。

然后执行三层清洗：

1. 删除 QueryRewrite 中与 Search-R1 train 或 test 的归一化原始问题相同的记录；
2. 保留之前已经完成 Search-R1 去污染的 clean V3.1；
3. 再次检查 QueryRewrite 与 clean V3.1 的原始问题交集，并从 QueryRewrite 侧删除交集。

本版本使用“纯问题文本匹配”，不依赖 `data_source`。这比只使用 `(data_source, question)` 更严格，可以避免同一个问题因为数据源字段不同而漏检。

Search-R1 参考数据为：

- [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)
- [Search-R1 官方代码仓库](https://github.com/PeterGriffinJin/Search-R1)

在 QueryRewrite 中，5,784 个问题被删除：其中 5,751 个只出现在 Search-R1 train，32 个只出现在 Search-R1 test，1 个同时出现在两者中。Search-R1 train/test 的问题交集按本次归一化规则已经统一处理。

## 为什么 QueryRewrite 变成了两条记录

QueryRewrite 原始记录的事件结构是：

```text
assistant: <think>...</think><search>...</search>
environment: <information>...</information>
assistant: <think>...</think><answer>...</answer>
```

为了和 V3.1 使用同一个 `prompt_text + target_text` 训练接口，每条保留的 QueryRewrite trajectory 被拆成两条标准化记录：

### `search_retention`

```text
prompt_text = 系统提示词 + 原问题
target_text = 第一个 assistant 的 think + search
```

### `final_answer`

```text
prompt_text = 系统提示词 + 原问题 + 第一个 think/search + information
target_text = 最终 think + answer
```

因此，`information` 只作为上下文输入，不作为 loss target；模型只对 assistant 的搜索动作和最终回答计算 loss。这和 V3.1 的前缀掩码训练目标是一致的。

## V3.1 的处理

V3.1 的原始训练文件没有直接保存 `question` 字段，因此通过 `source_id` 回溯到 V3.1 raw source 文件取得原始问题。

对 V3.1 执行了以下处理：

- 删除 Search-R1 train/test 中出现过的原始问题的全部 V3.1 记录；
- 对剩余问题保留全部 `search_retention` 记录；
- 对每个剩余 source trajectory 的两个重复 `final_answer_repair` 记录只保留一条。

所以 V3.1 的 975 个问题对应 975 条最终答案记录，以及 943 条 search-retention 记录。search-retention 数量没有强行补齐到 975，因为它原本是从候选 search turn 中全局采样得到的。

## JSONL 字段

每条最终记录都具有相同的字段：

| 字段 | 含义 |
|---|---|
| `id` | 最终记录的唯一 ID |
| `source_id` | 所属 trajectory 的 ID |
| `original_id` | 原始数据中的 ID |
| `source_dataset` | `queryrewrite` 或 `v3.1` |
| `data_source` | 原始数据源，例如 `triviaqa`、`musique` |
| `question` | 原始问题 |
| `question_normalized` | 用于去重的归一化问题 |
| `sample_type` | `search_retention`、`final_answer` 或 `final_answer_repair` |
| `prompt_text` | 输入上下文，训练时通常全部 mask |
| `target_text` | assistant 需要学习的输出 |
| `loss_mask_policy` | 当前为 `mask_prefix_only` |

最终文件可以直接交给读取 `prompt_text` 和 `target_text` 的 SFT 数据加载器。不要把它当作旧版带有 `events` 字段的 raw trajectory 文件，也不要直接用只接受 `events` 三元组的旧脚本读取它。

## 客观评价

这套数据可以称为“当前数据范围内干净、统一、可审计的 SFT 数据集”，但不应声称在绝对意义上“完美”。它的优点是：

- 已排除 Search-R1 train/test 的精确原始问题重合；
- QueryRewrite 与 V3.1 之间没有原始问题重合；
- V3.1 的重复 final-answer 记录已压缩为每个 trajectory 一条；
- 两类数据已经统一为相同的 prefix-target 训练接口；
- search 行为和最终回答的监督数量接近，分别为 5,159 和 5,191 条；
- 每条记录保留了来源数据集、原始 ID 和原始问题，便于审计。

仍然需要注意：

- 这里只能保证归一化后的精确问题不重合，不能保证语义改写、翻译、近义问题或共享证据文档不存在重合；
- QueryRewrite 占 8,432 条，V3.1 占 1,918 条，数据分布并不均衡，且 QueryRewrite 中 TriviaQA 占比较高；
- 当前文件是训练集，没有自动建立独立 validation split。后续切分必须按原始问题切分，不能把同一个问题的不同记录拆到 train 和 validation 两边；
- 上传 GitHub 前应确认 NQ、HotpotQA、TriviaQA、MuSiQue、2Wiki、WebQuestions 以及检索文档语料的许可证和再分发条款。

因此，推荐把本目录作为“清洗后的 SFT 数据发布目录”，训练前再单独建立 question-level validation split，并在训练脚本中检查 `prompt_text`、`target_text` 和最大序列长度。
