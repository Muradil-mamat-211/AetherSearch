# Multi-View RAGEN-2 × Exact-IG × A²TGPO Agentic RL 系统设计与逐 Update 工程实施报告

**文档版本：** V1.2（Update 1 / Update 2 / Update 11+ 分阶段执行版）  
**日期：** 2026-07-24  
**目标模型：** Qwen2.5-3B-DPO-V2 warm start  
**任务：** 多轮 Search Agent / Search-R1 风格开放域问答  
**训练原则：** 严格 rollout-start on-policy；`ppo_epochs=1`；每个成功 update 仅一次全局 `optimizer.step()`  
**框架：** verl + Ray + vLLM + FSDP2  
**服务器：**

- 5 × 48 GB GPU；
- 平台按每张 GPU 配置约 90 GB Host RAM 与 20 核 Xeon(R) Platinum 系列 CPU；
- 因而本报告按**约 450 GB Host RAM、100 个 CPU 核心**进行资源设计；
- 若云平台将“核”暴露为 SMT logical CPU，应以 `lscpu`、`ray.cluster_resources()` 与实际 NUMA 拓扑为最终事实，文中的 CPU 数量均视为 Ray `num_cpus` 初始配额。

**固定资源分工：**

- **GPU0：** Hybrid Retriever；
- **GPU1–4：** Agentic RL；
- **vLLM：** DP=4、TP=1，GPU1–4 每卡一个完整 Qwen2.5-3B rollout replica；
- **FSDP2：** world size=4，仅由 GPU1–4 组成；
- **GPU0 严禁加入 RL 的 NCCL communicator。**

> 本文严格区分：论文原始方法、项目扩展、工程优化和必须经 profile 决定的选项。训练开始前不能承诺 SOTA；是否达到领先水平必须由固定评测、多随机种子与消融实验验证。

---

# 0. 执行摘要

本项目把三条方法线组合为同一个同步 Agentic RL update：

1. **IGPO：** 用模型自身在相邻 Search 状态下生成 ground-truth answer 的条件 log-likelihood 增量，构造每个 Search turn 的 Exact Information Gain；
2. **RAGEN-2：** 用同一 Prompt 的 rollout reward variance 作为任务梯度 SNR 的代理，过滤低信号 Prompt；
3. **A²TGPO：** 对 IG 做同 `(prompt, search-index)` 的 turn-group normalization、方差重标定的未来 IG 累计，并形成 turn-level advantage。

本项目扩展：

- IG 与 Outcome 双通道 Prompt 筛选；
- channel robust scale 与 log-EMA；
- 前 10 个有效成功 update 的绝对健康基准；
- 两波 32 Prompt 形成 64 Prompt candidate pool；
- selected 少于 32 时 refill 32 Prompt，达到 96 后全局重筛；
- selected Prompt 约束为 32–36；
- Exact-IG 官方向量化 Fast Path + 独立 Prefix Oracle；
- GPU0 Retriever 与 GPU1–4 RL 隔离；
- vLLM/FSDP2 colocated 时分复用；
- 严格一次全局 optimizer step；
- 完整事务、日志、回滚、checkpoint 和性能剖析。

本文按三类 update 写清全部状态变化：

- **Update 1：** 初始化 scale，`b_1^c=m_1^c`；完整描述所有算法、计算、并行、Worker 与日志；
- **Update 2：** 使用已经提交的 `b_1^c` 进行选择，成功后第一次执行 EMA 更新得到 `b_2^c`；
- **Update 11+：** 若某通道已经积累 10 个有效成功观测，则启用绝对健康比例门；未满 10 个的通道继续使用 bootstrap gate。

---

# 1. 固定算法约束与统一符号

## 1.1 固定规模

\[
G=16
\]

\[
P_{\mathrm{wave}}=32,\qquad
P_{\mathrm{initial}}=64,\qquad
P_{\mathrm{max}}=96
\]

\[
P_{\mathrm{selected,min}}=32,\qquad
P_{\mathrm{selected,max}}=36
\]

\[
\rho_{\mathrm{Top-p}}=0.9
\]

初始 64 Prompt：

\[
64\times16=1024\ \text{trajectories}
\]

发生 refill：

\[
96\times16=1536\ \text{trajectories}
\]

## 1.2 模型角色

- \(\pi_\theta\)：当前可训练 Actor；
- \(\pi_{\mathrm{old},u}\)：本次 attempt 开始时冻结的 rollout/old-policy 快照；
- \(\pi_R\)：Exact-IG reward policy，本项目固定：
  \[
  \pi_R=\pi_{\mathrm{old},u};
  \]
- \(\pi_{\mathrm{ref}}\)：冻结的 DPO-V2 reference policy；
- vLLM rollout replica 是 \(\pi_{\mathrm{old},u}\) 的推理表示；
- FSDP2 Actor 是参数 source of truth。

## 1.3 数据索引

- \(p\)：Prompt；
- \(i\in\{1,\dots,G\}\)：Prompt \(p\) 的第 \(i\) 条 rollout；
- \(K_{p,i}\)：该 trajectory 的 Search turn 数；
- \(t\in\{0,\dots,K_{p,i}\}\)：状态前缀索引；
- \(h_{p,i,t}\)：完成第 \(t\) 次 `<search>→<information>` 后的完整上下文；
- \(\mathcal A_p\)：ground-truth aliases；
- \(y(a)\)：alias \(a\) 经冻结 GT schema 包装后的答案 token 序列。

## 1.4 Attempt 与 Update

- `attempt_id`：每次开始构建 candidate pool 就增加；
- `successful_update_step`：仅在一次全局 optimizer transaction 成功提交后增加；
- skip/abort 不推进：
  - Actor；
  - optimizer；
  - scheduler；
  - EMA；
  - health reference；
  - successful update step；
  - vLLM weight version。

---

# 2. 服务器、Ray Placement Group 与 CPU/内存预算

## 2.1 总资源假设

按用户提供的完整配置：

| 资源 | 总量 |
|---|---:|
| GPU | 5 × 48 GB |
| Host RAM | 约 450 GB |
| CPU | 约 100 核 |
| Retriever GPU | GPU0 |
| RL GPU | GPU1–4 |

建议只向 Ray 硬预留约 90 CPU，保留约 10 CPU 给：

- 操作系统；
- Ray GCS；
- NCCL/CUDA runtime；
- 文件系统与 checkpoint flush；
- 意外峰值和调试。

## 2.2 Ray 资源池

```text
RetrieverPool
    GPU0
    20 CPU
    custom resource: retriever_gpu=1

RLEnginePool
    GPU1–4
    4 bundles × (1 GPU + 6 CPU) = 24 CPU
    custom resource: rl_gpu=4

GlobalControlAndCPUWorkers
    46 CPU

Unreserved
    10 CPU
```

推荐 placement group：

```text
bundle_0: {"GPU": 1, "CPU": 20, "retriever_gpu": 1}
bundle_1: {"GPU": 1, "CPU": 6,  "rl_gpu": 1}
bundle_2: {"GPU": 1, "CPU": 6,  "rl_gpu": 1}
bundle_3: {"GPU": 1, "CPU": 6,  "rl_gpu": 1}
bundle_4: {"GPU": 1, "CPU": 6,  "rl_gpu": 1}
```

CPU-only actors使用剩余 46 个 Ray CPU。

## 2.3 CPU Worker 初始配额

| Worker / Actor | 数量 | 每个 `num_cpus` | 总 CPU | 作用 |
|---|---:|---:|---:|---|
| `RagenA2TGPOTrainer` | 1 | 2 | 2 | attempt 状态机、事务 |
| `PromptSamplerActor` | 1 | 2 | 2 | 采样、去重、cursor |
| `AgentLoopManager` | 1 | 4 | 4 | 32–64 个并发 coroutine |
| `OutcomeRewardWorker` | 8 | 2 | 16 | parser、EM/F1、format |
| `ExactIGTaskBuilder` | 1 | 8 | 8 | prefix、mask、position、packing |
| `CandidatePoolController` | 1 | 8 | 8 | variance、health、Top-p、advantage |
| `MetricsActor` | 1 | 2 | 2 | 异步指标聚合 |
| `CheckpointActor` | 1 | 2 | 2 | 分片状态与元数据写盘 |
| `FailureCoordinator` | 1 | 2 | 2 | timeout、abort、恢复 |
| **CPU-only 小计** |  |  | **46** |  |

Retriever 的 20 CPU 内部建议：

```text
HybridRetrieverCoordinator: 2 CPU
RetrieverBatchScheduler:    4 CPU
SparseBM25WorkerPool:       10 CPU
FusionAndDedupWorker:        2 CPU
Dense/FAISS GPU Actor host:  2 CPU
```

RLEnginePool 每个 GPU bundle 的 6 CPU 同时服务 colocated：

- vLLM server；
- FSDP2 rank；
- tokenizer/runtime；
- NCCL host thread。

vLLM 和 FSDP2 时分复用，因此不需要把二者 CPU 配额简单相加。

## 2.4 Host RAM 初始预算

| 用途 | 建议预算 |
|---|---:|
| OS / Ray GCS / page cache | 32 GB |
| Ray object store | 96 GB |
| Retriever、BM25、索引元数据 | 90 GB |
| RL worker heap、trajectory 与 pinned buffers | 180 GB |
| Checkpoint staging / 日志 | 32 GB |
| 安全余量 | 20 GB |
| **合计** | **450 GB** |

Ray object store 必须配置 NVMe spill 目录；不得让 1024/1536 条长 trajectory 的完整 Python 对象反复序列化复制。大对象通过 `ray.put()` 只存一次，Actor 间传 `ObjectRef`。

---

# 3. verl + Ray + vLLM + FSDP2 的控制拓扑

```text
RagenA2TGPOTrainer
├── PromptSamplerActor
├── AgentLoopManager
│   ├── AsyncVLLMServer GPU1
│   ├── AsyncVLLMServer GPU2
│   ├── AsyncVLLMServer GPU3
│   ├── AsyncVLLMServer GPU4
│   └── SearchAgentLoop coroutines
├── HybridRetrieverCoordinator
│   ├── DenseQueryEncoder GPU0
│   ├── FAISS GPU Index GPU0
│   ├── BM25 CPU Pool
│   └── Fusion/Dedup
├── OutcomeRewardWorker × 8
├── ExactIGTaskBuilder
├── FSDP2ActorWorkerGroup ranks 0–3 on GPU1–4
├── FSDP2Reference logical engine on GPU1–4
├── CandidatePoolController
├── MetricsActor
├── CheckpointActor
└── FailureCoordinator
```

## 3.1 vLLM

```text
DP = 4
TP = 1
GPU1–4 每卡一个完整模型 replica
```

每个 Prompt 的 16 条 trajectory 使用独立 request/seed；同一 trajectory 的后续 turn 使用 sticky session 固定在同一 vLLM server，同一 Prompt 的不同 trajectories 可分散至四个 replica。

## 3.2 FSDP2

```text
world_size = 4
ranks = GPU1, GPU2, GPU3, GPU4
```

默认目标：

```text
all FSDP modules:
    reshard_after_forward = false
```

作用：

- forward 前逐模块 all-gather；
- forward 后保留 unsharded 参数；
- learner backward 前不再重复 parameter all-gather；
- Exact-IG scoring window 的后续 micro-batch 可尽量复用 unsharded 参数。

必须记录实际峰值显存；若 learner 阶段 OOM，允许有版本化的 memory-balanced fallback，但不能静默改变。

## 3.3 vLLM 与 FSDP2 时分复用

```text
Rollout:
    vLLM awake
    FSDP2 idle/sharded

Exact IG:
    vLLM Level-1 sleep
    FSDP2 eval/inference

Selection:
    vLLM sleep
    FSDP2 reshard/idle
    CPU active

Refill:
    vLLM Level-1 wake
    FSDP2 idle

Learner:
    vLLM Level-2 sleep
    FSDP2 train
    Reference no-grad

Commit:
    FSDP2 new weights → vLLM
```

---

# 4. Update 1：完整算法与完整工程执行

Update 1 是整个项目的完整规范。Update 2 与 Update 11+ 只描述与此不同的状态更新。

---

## Update 1 / Step 0：BEGIN ATTEMPT 与状态冻结

### 目标

创建第一笔严格同步 on-policy 事务。

### Controller

- `RagenA2TGPOTrainer`：2 CPU；
- `FailureCoordinator`：2 CPU；
- FSDP2 rank/GPU 尚不执行计算。

### 固定状态

\[
\theta_{\mathrm{old},1}=\theta_0
\]

\[
\pi_R=\pi_{\mathrm{old},1}
\]

同时冻结：

```text
actor_weight_version
old_policy_version
reward_snapshot_version
reference_version
tokenizer_hash
chat_template_hash
gt_schema_hash
retriever_index_version
dense_encoder_version
prompt_sampler_rng
rollout_rng
```

### Update 1 特有状态

Update 1 尚无历史 scale：

\[
b_1^c\ \text{尚未提交}
\]

只有在最终 candidate pool 形成后，才计算：

\[
m_1^c=\operatorname{Median}\{e_p^c:e_p^c>0\}
\]

并临时设：

\[
b_{1,\mathrm{provisional}}^c=m_1^c.
\]

如果 attempt 最终 skip，这个 provisional scale 必须丢弃。

### 并行工作

`PromptSamplerActor` 可以同时预热：

- 数据文件句柄；
- dataset mixture alias table；
- 去重 Bloom/set；
- domain 配额；
- ground-truth alias cache。

### 必须记录

```text
attempt_id
successful_update_step=0
all model/version hashes
all RNG states
Ray placement group ids
GPU visibility map
FSDP world ranks
vLLM DP/TP
retriever version
BEGIN timestamp
```

### Gate

所有四个 vLLM replica 和四个 FSDP ranks 的 Actor checksum 必须一致，否则 abort。

---

## Update 1 / Step 1：抽取第一波 32 个 Prompt

### 工具与资源

- `PromptSamplerActor`：2 CPU；
- 可选利用其内部 2 个线程做数据读取和校验；
- 不使用 GPU。

### 计算与约束

从训练 mixture 中抽取：

\[
\mathcal C_1^{(1)}=\{p_1,\dots,p_{32}\}.
\]

必须完成：

- prompt_id 唯一；
- 不与当前 attempt 已采样 Prompt 重复；
- ground truth 非空；
- alias 规范化完成；
- 数据集/domain 配额符合 mixture；
- 最大初始 prompt token 长度符合 context budget；
- 样本具备可用 retriever query 环境。

### 可并行工作

在 32 Prompt 返回后：

- `ExactIGTaskBuilder` 可预先 tokenize 所有 aliases 与 GT schema；
- `OutcomeRewardWorker` 可预热 normalization/token-F1 规则；
- Retriever scheduler 可预热 dense encoder batch buffer。

### 输出

```text
PromptGroupSpec[32]
├── prompt_id
├── dataset/domain
├── prompt_token_ids
├── aliases
├── gt_schema_token_ids
├── sampling seeds[16]
└── expected context limits
```

### 必须记录

```text
sample_latency
dataset/domain histogram
prompt length p50/p95/p99
alias count distribution
duplicate_rejection_count
invalid_sample_rejection_count
data_cursor_before/after
sampler_rng_hash
```

### Gate

不足 32 个合法 Prompt 或数据游标错误，abort；不能复制 Prompt 补齐。

---

## Update 1 / Step 2：vLLM Rollout Wave 1（32 × 16 = 512）

### GPU 与 Engine

- GPU1–4：四个 `AsyncVLLMServer`；
- DP=4、TP=1；
- 每卡一个完整 Qwen2.5-3B；
- 每个 GPU bundle 分配 6 CPU 给 vLLM/FSDP colocated process；
- vLLM awake，FSDP2 idle/sharded。

### Agent 调度

- `AgentLoopManager`：4 CPU；
- 32–64 个并发 coroutine；
- 512 个 trajectory request 使用独立 seed；
- sticky session 保证一条 trajectory 的多轮生成始终落在同一 vLLM replica；
- load balancer 依据：
  - 当前 running request；
  - KV cache 使用；
  - 累计 generated tokens；
  - sticky session 负载。

### Retriever 并行

GPU0/20 CPU RetrieverPool 与 vLLM 同时运行：

```text
Search query
→ RetrieverBatchScheduler 2–5 ms 微批窗口
→ Dense encoder GPU0
→ FAISS batch search GPU0
|| BM25 CPU workers
→ Fusion/Dedup
→ <information> token IDs
→ 原 trajectory session
```

`<information>`：

- Attention 可见；
- policy loss mask=0；
- 不得 decode→rewrite→re-encode。

### 与 Rollout 同时异步执行的 CPU 工作

每条 trajectory 完成后立即将 ObjectRef 发给：

#### `OutcomeRewardWorker × 8`：16 CPU

计算：

- answer parser；
- terminal protocol；
- EM；
- multi-alias token-F1；
- valid/malformed/no-answer；
- format indicators；
- Search/query统计。

#### `ExactIGTaskBuilder`：8 CPU

准备：

- search prefix end positions；
- GT aliases；
- Fast Path structural mask metadata；
- logical position IDs；
- Oracle records；
- extended length；
- \(\sum L^2\) cost；
- multi-alias group index。

#### `MetricsActor`：2 CPU

流式记录 latency 和 token statistics，不阻塞 rollout。

### Outcome task reward

\[
R_{p,i}^{task}
=
\max_{a\in\mathcal A_p}
F1(\widehat a_{p,i},a)
\]

筛选通道中不加入 format reward。

### 终止格式指标

定义 terminal answer format indicator：

\[
F_{p,i}^{ans}
=
\mathbf 1[
\text{满足全部 terminal protocol 条件}
].
\]

全部条件：

1. 恰好一个闭合的 `<answer>...</answer>`；
2. `<answer>` 前后标签顺序合法；
3. 不存在未闭合 `<think>/<search>/<information>/<answer>`；
4. parser 成功定位 answer span；
5. `</answer>` 后除允许的 EOS/空白外没有模型文本；
6. trajectory 以 Answer 或版本化 fallback 正常终止。

该变量**不进入 RAGEN 的 \(V_p^O\)**，只用于后续 Answer-turn advantage。

### 每条 trajectory 必须持久化

```text
prompt_id, rollout_id, request_id, seed
prompt_ids, response_ids
response_mask
action_token_mask
environment_token_mask
padding_mask
turn_index_per_token
turn_action_spans
search_query_spans
information_spans
answer_span
search_prefix_end_positions
sampling config
vllm_weight_version
retriever_results
parser_status
termination_status
truncation_status
R_task
F_ans
```

### Wave 1 必须记录

```text
rollout_wave_1_seconds
trajectories_per_second
action_tokens_per_second
per-replica queue/running/waiting
KV cache usage
prefix cache hit rate
preemption count
retrieval query count
retrieval p50/p95/p99
parser success
format success
no_answer/malformed/truncation rate
search-turn distribution
repeat-query rate
GPU0–4 utilization and memory
CPU utilization by actor
Ray object store growth
```

### Gate

512 条 trajectory 必须全部有明确 terminal/timeout 状态。系统错误轨迹不能悄悄当成 wrong answer。

---

## Update 1 / Step 3：抽取第二波 32 个 Prompt

### 资源

- `PromptSamplerActor`：2 CPU。

### 时间优化

逻辑上这是 Step 3；工程上可在 Wave 1 后半段提前 prefetch，但只有在 Wave 1 Prompt IDs 已注册后才能提交，确保不重复。

### 输出

\[
\Delta\mathcal C_1^{(2)}=\{p_{33},\dots,p_{64}\}.
\]

与 Step 1 完全相同的校验、alias tokenize、seed 分配。

### 并行工作

此时 Wave 1 的：

- Outcome；
- parser；
- Exact-IG packing；
- Metrics 写入；

继续在 CPU 后台执行，不等待。

### 必须记录

与 Step 1 相同，额外记录：

```text
overlap_seconds_with_wave1_postprocessing
duplicate_against_wave1_rejections
```

---

## Update 1 / Step 4：vLLM Rollout Wave 2（再生成 512）

### 资源

与 Step 2 相同：

- GPU1–4 vLLM；
- GPU0 Retriever；
- AgentLoopManager 4 CPU；
- Outcome workers 16 CPU；
- ExactIGTaskBuilder 8 CPU。

### 最大并行化

```text
GPU1–4:
    Wave 2 rollout

GPU0 + Retriever CPU:
    Wave 2 search requests

OutcomeRewardWorker:
    完成 Wave 1 + 流式处理 Wave 2

ExactIGTaskBuilder:
    完成 Wave 1 masks/positions/packing
    + 流式准备 Wave 2

MetricsActor:
    写入 Wave 1 汇总 + Wave 2 实时数据
```

完成后：

\[
|\mathcal C_1^{(64)}|=64
\]

\[
N_{\mathrm{trajectory}}=1024.
\]

### 必须记录

与 Wave 1 相同，并记录两波差异：

```text
wave1_vs_wave2_throughput
wave1_vs_wave2_length
per-replica imbalance
CPU postprocessing backlog
retriever queue backlog
```

### Gate

确认每个 Prompt 恰有完整 \(G=16\) trajectories。不得用别的 Prompt 的 trajectory 填补不完整 group。

---

## Update 1 / Step 5：候选池 64 完整化与 vLLM→FSDP2 切换

### Controller

`RagenA2TGPOTrainer` 等待：

- 1024 trajectory 元数据齐全；
- Outcome 基础结果齐全或可追踪；
- IG task records 构建完成；
- 所有 object refs 可访问。

### vLLM

执行 Level-1 sleep：

- 释放 GPU KV cache；
- 释放/迁移推理权重所占 GPU 显存；
- 保留恢复同一 \(\theta_{\mathrm{old},1}\) 的能力；
- 因为之后可能 refill，不能丢失旧权重版本。

### FSDP2

GPU1–4 进入 Exact-IG scoring window：

```text
model.eval()
torch.inference_mode()
use_cache=False
all modules reshard_after_forward=False
```

### 切换期间可并行

CPU：

- Outcome workers 完成剩余解析；
- TaskBuilder 按 extended attention cost 排序；
- CandidatePoolController 预建 `(prompt, search-index)` peer map；
- MetricsActor 记录 sleep/unshard 延迟。

### 必须记录

```text
vllm_sleep_level
sleep_latency
released_gpu_memory
FSDP weight checksum
first_unshard_latency
IG task count
Oracle task count
Fast Path task count
extended length distribution
sum_L2 cost distribution
```

---

## Update 1 / Step 6：FSDP2 Exact-IG 64 与 Outcome 分支并行

## 6.1 Exact-IG 数学定义

对 alias \(a\)：

\[
\Phi_{p,i,t}(a)
=
\frac{1}{|y(a)|}
\sum_{j=1}^{|y(a)|}
\log\pi_R
\left(
y_j(a)\mid h_{p,i,t},y_{<j}(a)
\right)
\]

多 alias：

\[
\Phi_{p,i,t}
=
\max_{a\in\mathcal A_p}\Phi_{p,i,t}(a)
\]

Search turn IG：

\[
r_{p,i,t}^{IG}
=
\Phi_{p,i,t}-\Phi_{p,i,t-1}
\]

\[
\nabla_\theta r^{IG}=0.
\]

## 6.2 Fast Path

对完整 trajectory 追加所有状态对应的 GT 副本：

\[
[q,\tau_1,\dots,\tau_K,y^{(0)},\dots,y^{(K)}]
\]

使用：

- structural attention mask；
- padding attention mask；
- logical position IDs；
- GT score mask；
- shifted target mask；
- alias group index。

每个 \(y^{(t)}\) 只能看到：

\[
h_t+y_{<j}^{(t)}.
\]

RoPE position：

\[
\operatorname{pos}(y_j^{(t)})
=
\operatorname{pos}(h_t[-1])+j.
\]

## 6.3 Oracle Canary

默认：

```text
canary_fraction = 0.02
```

独立构造：

\[
[h_t;y(a)]
\]

检查：

\[
\Delta_\Phi,\qquad \Delta_{IG},
\]

IG sign agreement、\(V_p^{IG}\) 排序与 selected-set parity。

Fast Path 不支持、溢出或 parity 失败时自动回退 Oracle。

## 6.4 FSDP2 并行

- 四个 ranks 处理不同 Exact-IG records；
- dynamic batching 按 \(\sum L^2\) 平衡；
- 第一批逐模块 all-gather；
- 后续 micro-batch 尽量复用 unsharded 参数；
- 只返回 \(\Phi\) 标量，不长期保存完整 vocabulary logits。

## 6.5 同时进行的 CPU Outcome 分支

对于 Prompt \(p\)，有效 trajectories：

\[
\mathcal I_p^O=\{i:\mathrm{valid}_{p,i}=1\}
\]

\[
V_p^O
=
\operatorname{Var}_{i\in\mathcal I_p^O}
(R_{p,i}^{task}).
\]

若有效数少于 2，令 \(V_p^O=0\)。

Outcome workers/CandidatePoolController 可在 FSDP2 运行时完成：

\[
V_p^O,\quad
M_1^O,\quad
H_1^O,\quad
N_{1,O}^{+}
\]

但最终 \(S_p\)、Top-p 和 refill 决定必须等待 IG join。

## 6.6 IG 标量流式归约

FSDP2 每完成一个 batch，将 \(\Phi\) ObjectRef 发送给 CandidatePoolController。CPU 立即执行：

- multi-alias max；
- \(r^{IG}\)；
- per-turn peer 聚合；
- 已完成 Prompt 的 \(v_{p,t}^{IG}\)；
- \(V_p^{IG}\)。

### 必须记录

```text
Phi distribution by turn
rIG distribution by turn
positive/negative/zero IG rates
IG task throughput
effective tokens/s
attention cost units/s
all-gather time
peak allocated/reserved
Fast/Oracle max/P99 error
position-id failures
structural-mask failures
fallback rate
Outcome worker latency/backlog
VpO partial completion
```

### Gate

- NaN/Inf；
- Fast/Oracle 超阈值；
- position/mask 错误；
- FSDP collective 不一致；

任一出现，abort attempt 或回退并重新评分，不能继续使用不可信 IG。

---

## Update 1 / Step 7：计算 Prompt 级双通道信号

### 资源

- `CandidatePoolController`：8 CPU；
- NumPy/PyTorch CPU float64 vectorization；
- 不为每个 Prompt 创建 Ray task。

## 7.1 IG 通道

达到 Search index \(t\) 的 rollout 集合：

\[
\mathcal I_{p,t}^{IG}
=
\{i:K_{p,i}\ge t\},
\qquad
n_{p,t}=|\mathcal I_{p,t}^{IG}|.
\]

若 \(n_{p,t}\ge2\)：

\[
v_{p,t}^{IG}
=
\frac{1}{n_{p,t}-1}
\sum_{i\in\mathcal I_{p,t}^{IG}}
\left(
r_{p,i,t}^{IG}-\bar r_{p,t}^{IG}
\right)^2.
\]

自然支持权重：

\[
\omega_{p,t}
=
\frac{n_{p,t}}
{\sum_{k:n_{p,k}\ge2}n_{p,k}}.
\]

\[
V_p^{IG}
=
\sum_{t:n_{p,t}\ge2}
\omega_{p,t}v_{p,t}^{IG}.
\]

意义：

- \(v_{p,t}^{IG}\)：同 Prompt、同 Search 深度下，rollouts 的信息价值分歧；
- \(V_p^{IG}\)：Prompt 的 Search/process 学习信号强度；
- 不把不同 turn index 的结构性均值差误当方差。

## 7.2 Outcome 通道

\[
V_p^O
=
\operatorname{Var}_{i:\mathrm{valid}}
(R_{p,i}^{task}).
\]

意义：同一个 Prompt 的不同 rollouts 是否跨越“答对/答错或不同正确程度”的策略边界。

## 7.3 噪声地板

\[
e_p^c=[V_p^c-\nu_c]_+,
\qquad c\in\{IG,O\}.
\]

初始：

\[
\nu_c=10^{-12}
\]

但需由重复 scoring 噪声实测校准。

### 必须记录

```text
Vp,tIG
peer_count n_p,t
late-turn peer coverage
VpIG
VpO
e_p^IG/e_p^O
positive prompt count
insufficient peer flags
per-dataset/domain distributions
```

---

## Update 1 / Step 8：Update 1 Scale 初始化、通道激活与异质性

Update 1 没有历史 \(b^c\)。

## 8.1 最终候选池正中位数

在当前 64 pool 上先计算 provisional：

\[
m_{1,64}^c
=
\operatorname{Median}
\{e_p^c:e_p^c>0\}.
\]

若通道满足有效条件，临时：

\[
b_{1,\mathrm{provisional},64}^c=m_{1,64}^c.
\]

若后续 refill 到 96，必须废弃 64-pool provisional，并在最终 96 pool 重算。

## 8.2 绝对信号

\[
M_1^c
=
\frac1{P_c}\sum_{p=1}^{P_c}e_p^c.
\]

\[
N_{1,c}^{+}
=
\sum_p\mathbf1[e_p^c>0].
\]

## 8.3 Update 1 Bootstrap 激活

尚无 \(B_{\mathrm{ref}}^c\)，不能计算健康比例。

\[
a_1^c
=
\mathbf1[
M_1^c>0
\land M_1^c\text{ finite}
\land N_{1,c}^{+}\ge4
].
\]

通道独立，可能为：

\[
(a_1^{IG},a_1^O)
\in\{(1,1),(1,0),(0,1),(0,0)\}.
\]

## 8.4 尺度校准

若通道激活：

\[
U_p^c
=
\frac{e_p^c}
{b_{1,\mathrm{provisional}}^c+\epsilon}.
\]

关闭通道令其贡献为 0。

## 8.5 异质性

\[
H_1^c
=
\frac{\operatorname{Std}_p(e_p^c)}
{\operatorname{Mean}_p(e_p^c)+\epsilon}.
\]

意义：

- \(M_1^c\)：绝对信号强度；
- \(H_1^c\)：跨 Prompt 可分性；
- \(H\) 首版仅诊断，不作为硬 gate。

### 必须记录

```text
m_1_64^c
provisional_b_1_64^c
M_1^c
N_positive
a_1^c
H_1^c
U_p^c distribution
channel invalid reason
```

---

## Update 1 / Step 9：双通道融合与 64-Prompt Top-p

初始权重：

\[
\alpha_{IG}=\alpha_O=0.5.
\]

\[
S_p
=
\frac{
a_1^{IG}\alpha_{IG}U_p^{IG}
+
a_1^O\alpha_OU_p^O
}{
a_1^{IG}\alpha_{IG}
+
a_1^O\alpha_O+\epsilon
}.
\]

对正分数降序排序，选择最小 \(K^\star\)：

\[
\sum_{j=1}^{K^\star}S_{\sigma(j)}
\ge
0.9\sum_pS_p.
\]

分支：

1. \(32\le K^\star\le36\)：保留全部；
2. \(K^\star>36\)：保留最高 36；
3. \(K^\star<32\)：进入 refill；
4. 两通道均关闭：进入 refill；若最终仍关闭则 skip。

Top-36 后：

\[
\rho_{\mathrm{actual}}
=
\frac{\sum_{j=1}^{36}S_{\sigma(j)}}
{\sum_pS_p}.
\]

### 必须记录

```text
S_p per prompt
rank
K_star
selected ids
selected count
selected mass
rho_actual
channel tuple
selection by domain
zero-score count
refill decision reason
```

---

## Update 1 / Step 10A：若 selected<32，Refill 到 96

### Step 10A-1：FSDP→vLLM

- FSDP2 所有 ranks 显式 reshard；
- vLLM Level-1 wake；
- Actor 未更新，不需要权重同步；
- 验证 vLLM weight checksum 仍等于 \(\theta_{\mathrm{old},1}\)。

### Step 10A-2：采样新增 32 Prompt

- `PromptSamplerActor`：2 CPU；
- 不得与旧 64 Prompt 重复；
- 生成 32×16 seeds。

### Step 10A-3：Rollout 新增 512

与 Step 2 相同：

- vLLM GPU1–4；
- Retriever GPU0/20 CPU；
- Outcome workers 16 CPU；
- ExactIGTaskBuilder 8 CPU；
- AgentLoopManager 4 CPU。

### Step 10A-4：Exact-IG 只算新增 32

- vLLM Level-1 sleep；
- FSDP2 scoring；
- 旧64的 \(\Phi,r^{IG},V_p^{IG},V_p^O\) 全部复用；
- 只处理新增32。

### Step 10A-5：在完整 96 上重算 pool-level 统计

必须重算：

\[
M_1^c,\ H_1^c,\ N_{1,c}^{+},\ a_1^c,\ U_p^c,\ S_p
\]

以及全局 Top-p。

Update 1 特别注意：

\[
m_{1,96}^c
=
\operatorname{Median}_{p\in96}
\{e_p^c:e_p^c>0\}
\]

最终 provisional scale：

\[
b_{1,\mathrm{provisional}}^c=m_{1,96}^c.
\]

64-pool 的 provisional 值不得提交。

### Step 10A-6：仍不足 32

若：

\[
K^\star_{96}<32
\]

则 ABORT：

- 不 optimizer step；
- 不 scheduler step；
- 不提交 \(b_1^c\)；
- 不加入 health buffer；
- 不同步 vLLM；
- `attempt_id` 增加，`successful_update_step` 不变；
- 数据游标前进，从新的 Prompt 重新执行 Update 1 attempt。

### Refill 必须记录

```text
refill_used=1
wake/sleep/switch latency
new prompt ids
new rollout metrics
new Exact-IG metrics
cache reuse hit
final_pool_size=96
m_1_96
64-vs-96 health/heterogeneity
64-vs-96 selected-set change
final skip/success reason
```

---

## Update 1 / Step 10B：确定最终 selected Prompt groups

最终：

\[
32\le P_s\le36.
\]

对每个 selected Prompt 保留全部 \(G=16\) trajectories。

不得：

- 只保留高 reward rollout；
- 复制 Prompt；
- 从 nucleus 外补低分 Prompt；
- 在 selection 后重新生成 trajectory。

### selected 数据完整性

```text
P_s prompt groups
P_s × 16 trajectories
all raw IG
all outcomes
all action/environment masks
all turn spans
```

---

## Update 1 / Step 11：A²TGPO 标量 Advantage

### 资源与并行

- `CandidatePoolController`：8 CPU；
- 同时 GPU1–4 可开始 selected sequence 搬运、old-policy logprob no-grad scoring 的准备；
- 设置 barrier：advantage 与 old-logprob 均准备完成后才开始 learner backward。

## 11.1 Turn-group normalization

对同一 `(prompt, search-index)`：

\[
\widehat r_{p,i,t}^{IG}
=
\frac{
r_{p,i,t}^{IG}-\mu_{p,t}^{IG}
}{
\sigma_{p,t}^{IG}+\epsilon
}.
\]

若 peer 少于2或方差近零：

\[
\widehat r_{p,i,t}^{IG}=0.
\]

## 11.2 Outcome normalization

同一 Prompt 有效 trajectories：

\[
z_{p,i}^O
=
\frac{
R_{p,i}^{task}-\mu_p^O
}{
\sigma_p^O+\epsilon
}.
\]

无效 trajectory：

\[
z_{p,i}^O=0.
\]

## 11.3 未来 IG 累计

\[
D_{p,i,t}
=
\sum_{k=t}^{K_{p,i}}
\gamma^{k-t}
\widehat r_{p,i,k}^{IG},
\qquad \gamma=1.
\]

实际累计项数：

\[
n_{p,i,t}
=
\#\{k\ge t:\widehat r_{p,i,k}^{IG}\text{纳入累计}\}.
\]

## 11.4 Rescale

\[
\overline D_{p,i,t}
=
\frac{D_{p,i,t}}
{\sqrt{n_{p,i,t}}}.
\]

意义：归一化 IG 每项方差近似1时，累计和方差约随 \(n\) 增长；除以 \(\sqrt n\) 使不同深度 advantage 尺度近似可比。

## 11.5 明确定义 \(A_{p,i}^{format}\)

终端格式变量：

\[
F_{p,i}^{ans}\in\{0,1\}
\]

已在 rollout parser 中定义。

同 Prompt 中心化：

\[
\bar F_p^{ans}
=
\frac1G\sum_{j=1}^{G}F_{p,j}^{ans}.
\]

定义：

\[
\boxed{
A_{p,i}^{format}
=
F_{p,i}^{ans}-\bar F_p^{ans}
}
\]

性质：

- 有界于 \([-1,1]\)；
- 同一 Prompt 内均值为0；
- 全部格式正确时为0，不持续奖励已饱和格式；
- 全部格式错误时为0，避免仅靠统一负奖励破坏任务学习；
- 只作用在 Answer/fallback turn；
- 不进入 \(V_p^O\)、\(V_p^{IG}\) 或 Prompt selection。

若需要 malformed Search 局部惩罚，单独定义：

\[
A_{p,i,t}^{mal}
=
-\mathbf1[\text{该 Search turn parser 失败}],
\]

只作用于本 Search turn，系数 \(\lambda_{mal}\) 独立配置，不向未来或过去传播。

## 11.6 最终 Advantage

Search turn：

\[
A_{p,i,t}^{search}
=
\lambda_{IG}\overline D_{p,i,t}
+
\lambda_O z_{p,i}^{O}
+
\lambda_{mal}A_{p,i,t}^{mal}.
\]

Answer/fallback turn：

\[
A_{p,i}^{answer}
=
\lambda_Oz_{p,i}^{O}
+
\lambda_FA_{p,i}^{format}.
\]

初始：

\[
\lambda_{IG}=\lambda_O=\lambda_F=1.
\]

\(\lambda_{mal}\) 必须经 parser smoke test 后锁定；不能无记录地改变。

### 必须记录

```text
mu/sigma per (prompt,t)
normalized IG distribution
D_raw
n_accumulated
D_rescaled
variance by turn before/after rescale
z_outcome
F_ans
mean_F_ans
A_format
A_malformed
A_search
A_answer
zero-advantage rate
late-turn peer coverage
```

---

## Update 1 / Step 12：Old Logprob、Turn Ratio 与 Strict On-Policy Learner

## 12.1 Old logprob

由 rollout-start FSDP2 Actor 在 no-grad 下重算 selected action-token logprobs：

\[
\log\pi_{\mathrm{old},1}(a_k|h_k).
\]

vLLM sampled logprobs仅作 parity 诊断，不作为唯一 denominator。

## 12.2 Current policy forward

当前可微 logprob：

\[
\log\pi_\theta(a_k|h_k).
\]

在唯一 optimizer step 前：

\[
\theta=\theta_{\mathrm{old},1}.
\]

## 12.3 Turn ratio

\[
s_{p,i,t}(\theta)
=
\exp
\left[
\frac1{L_{p,i,t}}
\sum_{k\in\mathcal T_{p,i,t}}
\left(
\log\pi_\theta(a_k|h_k)
-
\log\pi_{\mathrm{old},1}(a_k|h_k)
\right)
\right].
\]

数值应：

\[
s_{p,i,t}\approx1
\]

但不能把它硬编码为1，因为 ratio 对当前 logprob 的梯度必须保留。

## 12.4 Adaptive clipping 审计接口

\[
c_{p,i,t}
=
1+\beta_c
[2\sigma(\widehat r_{p,i,t}^{IG})-1].
\]

\[
\psi_{p,i,t}
=
\min
\left[
s_{p,i,t}A_{p,i,t},
\operatorname{clip}
(
s_{p,i,t},
1-c_{p,i,t}\epsilon_{\mathrm{low}},
1+c_{p,i,t}\epsilon_{\mathrm{high}}
)
A_{p,i,t}
\right].
\]

严格单步 V1 中：

- ratio≈1；
- clip fraction≈0；
- 不宣称 adaptive clipping 产生实际约束；
- 仍记录公式结果以发现 engine mismatch 或错误 policy lag。

## 12.5 Loss Reduction

固定：

\[
J_{\mathrm{policy}}
=
\frac1{P_s}
\sum_{p=1}^{P_s}
\frac1G
\sum_{i=1}^G
\frac{
\sum_tL_{p,i,t}\psi_{p,i,t}
}{
\sum_tL_{p,i,t}
}.
\]

即：

```text
Prompt mean
→ Trajectory mean
→ trajectory 内 action-token mean
```

## 12.6 Reference KL

Canonical：

\[
L
=
-J_{\mathrm{policy}}
+
\beta_{\mathrm{KL}}L_{\mathrm{KL}},
\qquad
\beta_{\mathrm{KL}}=0.01.
\]

只在 action tokens 上计算，`<information>` 不参与。Full-vocabulary KL 使用 token/vocab chunk，禁止长期保存 Actor 与 Reference 的全部 logits。

## 12.7 Prompt-group 分布式打包

selected 32–36 Prompt 以完整 group 为原子，按 action-token 负载分配到四个 ranks。

不能简单 local mean 后再平均；每个 trajectory 的权重必须精确为：

\[
\frac1{P_sG}
\]

其内部每个 action token 权重为：

\[
\frac1{P_sG\,N_{p,i}^{action}}.
\]

### 必须记录

```text
old/current logprob parity
turn ratio mean/std/min/max
clip bounds
clip fraction
policy loss
KL loss
entropy
per-rank prompt groups
per-rank action tokens
load imbalance
full-vocab KL chunk stats
```

---

## Update 1 / Step 13：唯一一次全局 Optimizer Transaction

### 严格执行顺序

```text
zero_grad once

for each gradient-accumulation micro-batch:
    reference no-grad forward/chunked KL
    actor forward
    compute turn ratio/surrogate
    apply exact global sample weights
    backward
    NO optimizer.step
    NO scheduler.step
    NO actor mutation

after all micro-batches:
    global grad norm
    gradient clipping
    optimizer.step once
    scheduler.step once
```

必须验证每两个 micro-batch 之间 Actor checksum 不变。

### FSDP2

- all modules `reshard_after_forward=False`；
- backward 复用 forward 的 unsharded 参数；
- gradient reduce-scatter；
- 每个 rank 更新自己的 parameter/optimizer shard；
- 所有 ranks collective 顺序一致。

### 必须记录

```text
gradient_accumulation_microbatch_count
microbatch prompt/action-token counts
parameter checksum before each microbatch
gradient norm
grad clipping fraction
reduce-scatter time
optimizer step time
scheduler step count
optimizer_steps_in_attempt=1
peak GPU memory
```

### Gate

以下任一不满足即 abort/rollback：

- `optimizer_steps_in_attempt != 1`；
- micro-batch 间 checksum 改变；
- NaN/Inf；
- collective mismatch；
- ratio 非预期偏离；
- clip fraction异常大；
- global normalization错误。

---

## Update 1 / Step 14：COMMIT Update 1

只有 learner transaction 成功后提交。

## 14.1 Scale 初始化提交

从**最终 candidate pool**（64或96）计算的：

\[
m_1^c
\]

提交为：

\[
\boxed{b_1^c=m_1^c}.
\]

仅对有效通道提交。无有效正中位数的通道保持“未初始化”。

## 14.2 Health bootstrap

对每个有效通道保存：

\[
M_1^c.
\]

该通道 bootstrap count 增加1。

尚不计算正式：

\[
B_{\mathrm{ref}}^c.
\]

## 14.3 vLLM 权重同步

FSDP2 新 Actor：

\[
\theta_1
\]

同步到四个 vLLM replica。记录 per-replica checksum，全部一致后才允许 Update 2 rollout。

## 14.4 Checkpoint

提交：

```text
actor sharded state
optimizer state
scheduler
b_1^c
health buffers
bootstrap counts
successful_update_step=1
attempt_id
data cursor
all RNG
all model/config hashes
retriever version
```

### 必须记录

```text
COMMIT timestamp
b_1^IG/b_1^O
M_1^IG/M_1^O
bootstrap counts
weight sync latency
vLLM checksums
checkpoint duration
wall clock per successful update
```

---

# 5. Update 2：与 Update 1 的区别

Update 2 的 rollout、Exact-IG、RAGEN、refill、A²TGPO、strict learner 和日志步骤与 Update 1 完全相同。以下仅列状态差异。

## 5.1 Update 2 进入状态

\[
\theta_{\mathrm{old},2}=\theta_1.
\]

对于已在 Update 1 成功初始化的通道，Update 2 selection 使用固定的历史 scale：

\[
\boxed{
b_{\mathrm{use},2}^c=b_1^c
}
\]

不能使用当前 Update 2 的 \(m_2^c\) 给自身即时归一化。

未初始化的通道仍按 Update 1 provisional 规则处理；两个通道独立。

## 5.2 Update 2 通道激活

通道尚未收集满10个有效成功观测，因此仍使用 bootstrap gate：

\[
a_2^c
=
\mathbf1[
M_2^c>0
\land M_2^c\text{ finite}
\land N_{2,c}^{+}\ge4
].
\]

不使用健康比例 \(h_2^c\)。

## 5.3 Update 2 的 \(U_p^c\)

已初始化通道：

\[
U_p^c
=
\frac{e_p^c}{b_1^c+\epsilon}.
\]

注意：本 update 内发生 64→96 refill 时，\(b_1^c\) 始终固定。

## 5.4 第一次 EMA 更新

最终 candidate pool 得到：

\[
m_2^c
=
\operatorname{Median}\{e_p^c:e_p^c>0\}.
\]

只有 Update 2 成功 optimizer commit 后：

\[
\boxed{
\log b_2^c
=
(1-\eta)\log b_1^c+\eta\log m_2^c
}
\]

\[
\eta
=
1-2^{-1/10}
\approx0.066967.
\]

\(b_2^c\) 用于 Update 3，而不是反过来影响 Update 2 selection。

若通道无效或关闭：

- 冻结该通道 \(b^c\)；
- 不追加该通道的 health observation；
- 其他通道可正常更新。

## 5.5 Health buffer

成功后追加有效的：

\[
M_2^c.
\]

若这是该通道第2个有效成功观测，bootstrap count=2。不同通道的 count 可以不同。

## 5.6 Update 2 额外日志

```text
b_used_2^c=b_1^c
m_2^c
EMA log-space delta
b_2^c proposal/commit
scale freeze reason
health bootstrap count per channel
```

---

# 6. Update 11+：绝对健康门启用阶段

这里的“Update 11”指第11个成功 policy update。健康门按**通道独立有效观测数量**启用，而不是机械按全局 update 编号。

## 6.1 Health reference

若通道 \(c\) 已收集10个有效成功观测：

\[
\{M_{(1)}^c,\dots,M_{(10)}^c\},
\]

则冻结初始健康参考：

\[
\boxed{
B_{\mathrm{ref}}^c
=
\operatorname{Median}
\{M_{(1)}^c,\dots,M_{(10)}^c\}.
}
\]

括号索引表示该通道的有效观测序号。

若某通道到全局 Update 11 仍未满10个有效观测，它继续使用 bootstrap gate，不能伪造 \(B_{\mathrm{ref}}^c\)。

## 6.2 Update 11 进入 scale

假设前面均有效，Update 11 selection 使用：

\[
b_{\mathrm{use},11}^c=b_{10}^c.
\]

本 update 的 \(m_{11}^c\) 只在成功 commit 后形成 \(b_{11}^c\)。

## 6.3 绝对健康度

最终 candidate pool：

\[
M_{11}^c
=
\frac1{P_c}\sum_pe_p^c.
\]

\[
h_{11}^c
=
\frac{M_{11}^c}
{B_{\mathrm{ref}}^c+\epsilon}.
\]

正式激活：

\[
\boxed{
a_{11}^c
=
\mathbf1[
h_{11}^c\ge0.1
\land N_{11,c}^{+}\ge4
].
}
\]

意义：

- \(h\ge0.1\)：当前绝对信号至少保持初始健康参考的10%；
- `N_positive≥4`：避免仅由极少数异常 Prompt 支撑通道。

## 6.4 异质性仍非硬 Gate

\[
H_{11}^c
=
\frac{\operatorname{Std}_p(e_p^c)}
{\operatorname{Mean}_p(e_p^c)+\epsilon}.
\]

- Health 低、H 高：仍关闭通道；
- Health 高、H 低：通道可更新，但 Top-p 的边际价值可能较小；
- 首版不动态改 \(\rho\)，只记录供消融。

## 6.5 Update 11 的 EMA

若通道健康、有效且 update 成功：

\[
\log b_{11}^c
=
(1-\eta)\log b_{10}^c
+
\eta\log m_{11}^c.
\]

若通道被健康门关闭：

- 冻结 scale；
- 不让低信号池把 EMA 拉向0；
- 记录 `EMA_FREEZE_LOW_HEALTH`。

## 6.6 两通道均关闭

64 pool 两通道均关闭：

- 可 refill 到96；
- 96仍关闭：skip；
- 不 optimizer；
- 不 EMA；
- 不 scheduler；
- 可增加 consecutive low-health counter；
- 连续达到版本化阈值后触发 early-stop review，而不是自动改变算法。

## 6.7 Update 11 额外日志

```text
B_ref^c
M_11^c
h_11^c
health threshold
N_positive
a_11^c
bootstrap-vs-health-gate mode
EMA update/freeze
consecutive low-health count
```

---

# 7. 每一步统一的结构化日志

## 7.1 Attempt/Update

```text
attempt_id
successful_update_step
status
skip_reason
candidate_pool_size
selected_prompt_count
selected_trajectory_count
refill_used
wall_clock
optimizer_steps_in_attempt
```

## 7.2 Prompt

```text
prompt_id
dataset/domain
VpIG
VpO
eIG/eO
UIG/UO
S
rank
selected
selection reason
outcome distribution
search-count distribution
```

## 7.3 Trajectory

```text
prompt_id/rollout_id
seed/request_id
token lengths
turns
queries
retrieved ids
R_task
F_ans
Phi list
IG list
validity
```

## 7.4 Turn

```text
turn index
action span
query
retrieved ids
raw IG
normalized IG
n_accumulated
D_raw/D_rescaled
z_outcome
A_format/A_malformed
final advantage
ratio
clip scale
clipped
```

## 7.5 系统性能

```text
GPU0–4 utilization/memory
vLLM queues/KV/preemption
FSDP all-gather/reduce-scatter
CPU actor utilization
Ray object store/spill
retriever latency
sleep/wake/weight-sync
checkpoint duration
```

---

# 8. 关键上线 Gate

1. Fast/Oracle Exact-IG parity 达标；
2. position IDs、structural mask、causal shift正确；
3. `<information>` Attention 可见但 policy mask=0；
4. vLLM DP=4/TP=1 weight checksum一致；
5. GPU0 不属于 FSDP/NCCL world；
6. selected 32–36 的 Prompt/trajectory/token权重正确；
7. refill 只计算新增32；
8. Update 1 `b_1^c=m_1^c` 仅成功后提交；
9. Update 2 EMA 为滞后一拍，不能当前池自归一化；
10. Update 11 health gate 按通道独立有效计数；
11. micro-batch 间 Actor checksum不变；
12. `optimizer_steps_in_attempt=1`；
13. ratio≈1、clip fraction≈0；
14. skip不改变 optimizer/scheduler/EMA；
15. checkpoint恢复后所有 scale、health、RNG 连续。

---

# 9. 建议配置骨架

```yaml
hardware:
  total_gpus: 5
  gpu_memory_gb: 48
  host_ram_gb_approx: 450
  cpu_cores_approx: 100
  retriever_gpu_ids: [0]
  rl_gpu_ids: [1, 2, 3, 4]
  cpu_reserved_for_os: 10
  ray_object_store_gb: 96

ray:
  retriever_pool_cpus: 20
  rl_engine_cpus_per_gpu: 6
  controller_cpu_workers: 46

algorithm:
  group_size: 16

  candidate_pool:
    rollout_wave_prompts: 32
    initial_prompts: 64
    refill_prompts: 32
    max_prompts: 96

  selection:
    top_p_mass: 0.90
    include_zero: false
    min_selected_prompts: 32
    max_selected_prompts: 36
    alpha_ig: 0.5
    alpha_outcome: 0.5
    noise_floor_ig: 1.0e-12
    noise_floor_outcome: 1.0e-12

  scale:
    ema_half_life: 10
    health_reference_valid_updates: 10
    health_threshold_ratio: 0.10
    minimum_positive_prompts: 4
    update1_initialize_b_from_m: true
    lagged_scale_after_update1: true
    freeze_scale_when_channel_inactive: true

  exact_ig:
    reward_policy: rollout_start_actor
    default_path: vectorized
    oracle_enabled: true
    canary_fraction: 0.02
    multi_alias_reduce: max
    dynamic_batch_cost: attention_quadratic
    use_cache: false

  a2tgpo:
    gamma: 1.0
    lambda_ig: 1.0
    lambda_outcome: 1.0
    lambda_format: 1.0
    lambda_malformed_search: null
    format_advantage: centered_binary_within_prompt
    ppo_epochs: 1
    strict_on_policy: true
    optimizer_mini_steps: 1
    gradient_accumulation_over_selected_batch: true

  kl:
    coefficient: 0.01
    target_mode: full_vocab
    action_tokens_only: true

actor_rollout_ref:
  actor:
    strategy: fsdp2
    fsdp_config:
      reshard_after_forward: false
      param_offload: false
      optimizer_offload: false

  ref:
    strategy: fsdp2
    fsdp_config:
      reshard_after_forward: true

  rollout:
    name: vllm
    data_parallel_size: 4
    tensor_model_parallel_size: 1
    enable_sleep_mode: true
    enable_prefix_caching: true
    enable_chunked_prefill: true
    max_num_seqs: 32
    gpu_memory_utilization: 0.35

reward:
  outcome_workers: 8
  outcome_worker_cpus: 2
  stream_with_agent_loop: true

trainer:
  advance_scheduler_on_skip: false
  checkpoint_on_successful_update_only: true
```

该 YAML 是项目逻辑 schema；实际字段必须根据服务器上的 verl commit 进行只读审计和映射。

---

# 10. Codex 实施顺序

1. 只读审计实际 verl/Ray/vLLM/FSDP2 版本与扩展点；
2. 实现纯算法单元与手工数组测试；
3. Exact-IG Oracle/Fast Path parity；
4. AgentLoop TITO 与 GPU0 Retriever；
5. 2 Prompt × G2 无 optimizer；
6. 4 Prompt × G4 strict one-step；
7. 64 Prompt × G16 profile；
8. 64→96 refill；
9. Update 1 scale bootstrap；
10. Update 2 EMA；
11. Update 11 health gate；
12. 10–20 update pilot；
13. 固定 eval gate；
14. 正式训练。

---

# 11. 参考资料

1. Wang et al., *Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents*, arXiv:2510.14967.
2. Wang et al., *RAGEN-2: Reasoning Collapse in Agentic RL*, arXiv:2604.06268.
3. Chen et al., *A²TGPO: Agentic Turn-Group Policy Optimization with Adaptive Turn-level Clipping*, arXiv:2605.06200.
4. PyTorch FSDP2 official documentation.
5. verl HybridFlow、Agent Loop、Reward Loop、Engine Workers documentation.
6. vLLM Sleep Mode and Paged KV Cache documentation.
7. Ray placement group and resource scheduling documentation.


---

# 12. 完整模型质量、Reasoning Collapse 与评测指标

以下指标不必在每个 GPU kernel 后即时阻塞计算，但必须按 update 或固定 checkpoint 周期异步汇总。

## 12.1 Rollout 行为

```text
format_success_rate
parser_success_rate
answer_rate
no_answer_rate
malformed_search_rate
malformed_answer_rate
truncation_rate
max_turn_termination_rate
avg_search
search_turn_count_histogram
multi_search_rate
repeat_query_rate
unique_query_rate
query_edit_distance
query_semantic_similarity
reasoning/action/environment token lengths
```

## 12.2 Reward 与信用分配

```text
task_f1 / EM
R_task mean/std/min/max
Phi_t by turn-index
rIG by turn-index
positive/negative/zero IG rate
Vp,tIG
VpIG
VpO
late-turn peer coverage
normalized IG variance by turn
D_raw variance by turn
D_rescaled variance by turn
A_search / A_answer / A_format distributions
```

## 12.3 RAGEN-2 / SNR

```text
M_IG / M_O
H_IG / H_O
b_IG / b_O
m_IG / m_O
B_ref_IG / B_ref_O
h_IG / h_O
N_positive
channel activation tuple
S_p distribution
Top-p K*
rho_actual
candidate-to-selected retention
selected domain composition
selection overlap across updates
consecutive low-health count
```

## 12.4 Policy Optimization

```text
policy_loss
reference_KL
beta_KL
entropy
old/current/vLLM logprob parity
turn ratio
clip fraction（严格 V1 预期≈0）
gradient norm
gradient clipping fraction
learning rate
optimizer/scheduler step counters
parameter update norm
```

## 12.5 Reasoning Collapse

RAGEN-2 指出 entropy 不能单独判断模型是否仍响应输入，因此同时监控：

```text
within-prompt rollout entropy
cross-prompt MI proxy
input-conditioned query distinguishability
template similarity
reasoning prefix similarity
search-query diversity
answer/reasoning mutual dependence proxy
reward variance cliff
gradient spike
```

报警示例：

- entropy 稳定但 MI proxy 快速下降；
- \(M^c\) 低于初始基准10%；
- multi-search 下降而单次 Search 模板相似度升高；
- selected Prompt 长期集中于单一数据集或答案类型。

## 12.6 固定评测

至少包括：

```text
NQ
TriviaQA
PopQA
HotpotQA
2WikiMultiHopQA
MuSiQue
Bamboogle
single-hop aggregate
multi-hop aggregate
```

固定记录：

```text
EM/F1
avg_search
multi_search_rate
repeat_query_rate
format rate
context length
retrieval success proxy
```

建议 checkpoint_0、每 50 个成功 update 和训练结束执行同一 Eval-1000；最终报告至少提供 3 个随机种子的均值和标准差。

---

# 13. 完整 Checkpoint 状态

成功 update 的 checkpoint 必须保存：

```text
actor FSDP2 sharded state
optimizer sharded state
scheduler state
successful_update_step
attempt_id
data cursor
dataset mixture state
prompt sampler RNG
Python RNG
NumPy RNG
PyTorch CPU/CUDA RNG
vLLM sampling seed/version
b_IG / b_O
EMA half-life and eta
health bootstrap buffers
health valid-count per channel
B_ref_IG / B_ref_O
consecutive low-health/skip counters
candidate/refill parameters
selection parameters
A2TGPO parameters
KL parameters
GT schema/token ids/hash
tokenizer/chat-template hash
reference-model hash
retriever index/hash/version
dense encoder hash
verl commit
Ray version/config
vLLM version
PyTorch/Transformers versions
attention backend
FSDP2 wrap policy
reshard_after_forward configuration
algorithm source-code commit
```

默认不保存未提交 attempt 的临时候选池。节点故障后：

1. abort 当前 attempt；
2. 从最后成功 update 恢复；
3. 检查 FSDP/vLLM checksum；
4. 重新开始新的 candidate pool。

---

# 14. 测试矩阵与正式上线门禁

## 14.1 单元测试

- multi-alias EM/F1；
- answer parser 与 \(F^{ans}\)；
- malformed Search 局部标记；
- TITO 与 token provenance；
- causal shift；
- structural attention mask；
- position IDs / RoPE；
- Fast/Oracle \(\Phi\)；
- \(r^{IG}\)；
- \(v_{p,t}^{IG},V_p^{IG},V_p^O\)；
- Update 1 provisional scale；
- Update 2 lagged EMA；
- Update 11 channel gate；
- Top-p/refill/skip；
- A²TGPO normalization、rescale；
- \(A^{format}\)；
- strict ratio gradient；
- prompt→trajectory→token reduction；
- single optimizer transaction；
- checkpoint/resume。

## 14.2 分布式测试

- 4 FSDP ranks collective 次数一致；
- selected=32/33/34/35/36；
- 每 rank token-aware 不均衡 batch；
- 空 micro-batch 使用 zero-loss dummy；
- GPU0 未进入 RL NCCL group；
- vLLM DP=4/TP=1；
- sleep/wake/checksum；
- worker timeout/abort；
- Ray object store spill；
- checkpoint restore。

## 14.3 Smoke 路径

```text
Stage A: 2 prompts × G2，无 optimizer
Stage B: 4 prompts × G4，一次 optimizer
Stage C: 8 prompts × G4，Exact-IG 全量 parity
Stage D: 64 prompts × G16，只 profile，不保存 checkpoint
Stage E: 64→96 refill
Stage F: Update 1 bootstrap
Stage G: Update 2 EMA
Stage H: 构造/回放 Update 11 health gate
Stage I: 10–20 个成功 update pilot
Stage J: 固定 eval gate 后正式训练
```

## 14.4 硬验收

- Exact-IG parity 达标；
- no token/span/mask 错位；
- `optimizer_steps_in_attempt=1`；
- micro-batch 间参数不变；
- ratio≈1；
- clip fraction≈0；
- skip事务无状态泄漏；
- EMA 与 health reference 可恢复；
- 64→96 只计算新增32；
- CPU/GPU无长期空闲关键路径；
- Retriever p95 不成为 rollout 主瓶颈；
- multi-search 和 multi-hop 不再出现已知快速坍缩。

---

# 15. 性能 Profile 与自适应工程决策

前 20–30 个 attempt 必须测量：

```text
T_sample_32
T_rollout_32
T_retrieval
T_outcome
T_ig_prep
T_vllm_sleep/wake
T_exact_ig_64
T_exact_ig_32_refill
T_selection
T_advantage
T_old_logprob
T_reference_KL
T_actor_forward_backward
T_optimizer
T_weight_sync
T_checkpoint
refill_rate
```

## 15.1 64→96 是否保留

若 refill rate 接近100%，且：

\[
T_{\mathrm{switch}}+T_{\mathrm{refill32}}+T_{\mathrm{IG32}}
\]

长期高于一次性生成96带来的额外成本，则可形成版本化工程实验：

```text
candidate_mode=96_upfront
```

但这属于 profile 后的工程改动，不能静默改变算法版本。

## 15.2 CPU 动态调整

- Outcome backlog 高：增加 worker concurrency，但总 CPU 不超过预留；
- IG task prep 落后于 rollout：从 CandidateController 暂借 CPU 给 TaskBuilder；
- Retriever queue 高：将更多 CPU 临时给 BM25/fusion；
- CandidatePool 数学计算只处理64/96个数，不应成为瓶颈；若占时异常，优先检查序列化而非继续加 CPU。

## 15.3 GPU 优化顺序

1. vLLM request batching 与 sticky balancing；
2. Retriever query micro-batching；
3. Exact-IG Fast Path parity；
4. \(\sum L^2\) packing；
5. FSDP2 no-reshard scoring window；
6. learner no-reshard；
7. full-vocab KL chunk/fusion；
8. weight sync；
9. allocator fragmentation。

---

# 16. 最终技术结论

本 V1.2 报告将整个项目按 Update 1、Update 2、Update 11+ 明确拆分：

\[
\boxed{
\text{Update 1：从最终池初始化 }b_1^c=m_1^c
}
\]

\[
\boxed{
\text{Update 2：使用 }b_1^c\text{ 筛选，成功后开始 EMA 得到 }b_2^c
}
\]

\[
\boxed{
\text{Update 11+：满足每通道10个有效观测后启用绝对健康门}
}
\]

同时固定：

\[
\boxed{
\text{GPU0 Retriever；GPU1–4 vLLM DP4/TP1 与 FSDP2 world-size4}
}
\]

\[
\boxed{
\text{CPU Outcome/IG-prep 与 GPU rollout/Exact-IG 最大化重叠}
}
\]

\[
\boxed{
\text{selected 32–36 Prompt groups，完整保留每组 G=16}
}
\]

\[
\boxed{
\texttt{ppo_epochs=1,\ optimizer\_mini\_steps=1}
}
\]

因此该版本既完整呈现算法，也给出了每一步的 Worker、CPU/GPU 配额、并行窗口、数据输出、日志与失败 gate，可作为 Codex 后续只读审计和分阶段实现的正式 System Design Specification。
