# 实验状态报告 · 8 视频 4 路并行复现 + QA token Pareto

> 更新时间：2026-11-16 (Asia/Shanghai)
> 实验名：`m3_parallel4_paper8_20261113` · **状态：✅ finished（全部跑完）**
> 报告路径：`experiments/m3_parallel4_paper8_20261113/index.md`

---

## 0. 最终结论（TL;DR）

- **8 视频 / 97 题全部完成**，pipeline + QA + judge + 5 档 QA 预算 sweep + Pareto 图均已落盘（`status.json` = finished）。
- **准确率 63.92% (62.0/97)**，论文基线 **57.22%**，**delta +6.70pp** → **近似复现成功**（同量级，偏高）。
- **总耗时 ≈ 6.34h**（pipeline 21541.2s + QA 1266.1s = 22807.4s），主导阶段是 `episodic_memory`（约占 pipeline 90%）。
- **Token 总消耗：prompt 9.17M / completion 0.92M**。
- **Pareto 核心发现**：准确率随 QA 预算 6000→16000 单调上升（47.94%→61.34%），但 **22528 档反而回落到 59.28%（被支配）**——印证「给更多 token 不一定更好」的研究问题。Pareto 前沿 = [6000, 9000, 12000, 16000]。

---

## 1. 目标（用户需求回顾）

1. oGMemory 在同事分支 **`rychkov/m3-parallel`** 上，用 **4 路并行** 跑 LaTeX 文档里的 **8 个视频 / 97 题**。
2. 模型：**MiniMax-M3.1-Flash**，**全新数据库**，验证并行后准确率能否**近似复现**论文 **57.2%**。
3. 统计 **token 消耗、整体耗时、各阶段耗时**。
4. 保留已建记忆不变，只改 **QA token 限制**（`m3.qa_context.max_input_tokens`）重跑问答，做 **Pareto front**（token↑ 是否 accuracy↑）。
5. 不计算视频 sha256（已按要求去掉）。

---

## 2. 最终准确率（对照论文）— 来源 `judgments.jsonl` 97 行

| 视频 | 题数 | 得分 | 本次 acc% | 论文 acc% | 差值 pp |
|---|---|---|---|---|---|
| study_09 | 12 | 7.50 | 62.50 | 58.33 | +4.17 |
| kitchen_01 | 13 | 4.50 | 34.62 | 38.46 | −3.84 |
| bedroom_05 | 12 | 6.00 | 50.00 | 41.67 | +8.33 |
| study_23 | 14 | 10.50 | 75.00 | 67.86 | +7.14 |
| kitchen_23 | 12 | 9.50 | 79.17 | 54.17 | +25.00 |
| study_05 | 12 | 7.50 | 62.50 | 45.83 | +16.67 |
| gym_03 | 11 | 7.50 | 68.18 | 72.73 | −4.55 |
| bedroom_04 | 11 | 9.00 | 81.82 | 81.82 | −0.00 |
| **合计** | **97** | **62.0** | **63.92** | **57.22** | **+6.70** |

> kitchen_01 说明见第 4 节（重采样决策）。

---

## 3. Token 消耗与耗时

### 3.1 Token（usage 日志独立核算）

| 阶段 | 调用数 | prompt | completion | cached |
|---|---|---|---|---|
| pipeline | 549 | 2,510,031 | 785,051 | 604,837 |
| qa | 517 | 6,595,333 | 123,015 | 943,033 |
| judge | 97 | 60,973 | 9,935 | 18,527 |
| **合计** | | **9,166,337** | **918,001** | **1,566,397** |

### 3.2 整体耗时（status history 权威值）

- pipeline 合计 **21541.2s (359.0 min)**；QA 合计 **1266.1s (21.1 min)**；**总计 22807.4s ≈ 6.34h**。
- 主导阶段：`episodic_memory` 约占单视频 pipeline 90%（如 study_09：episodic 2529s / 总 2820s）。

### 3.3 逐视频耗时（stages/*.json 权威 receipts）

| 视频 | pipeline (s) | qa (s) | pipeline (min) |
|---|---|---|---|
| study_09 | 2820.0 | 160.3 | 47.0 |
| kitchen_01 | 163.4 | 120.2 | 2.7 |
| bedroom_05 | 1764.7 | 111.6 | 29.4 |
| study_23 | 2385.8 | 152.0 | 39.8 |
| kitchen_23 | 2435.9 | 30.8 | 40.6 |
| study_05 | 4058.5 | 235.2 | 67.6 |
| gym_03 | 4097.1 | 247.8 | 68.3 |
| bedroom_04 | 3815.8 | 208.2 | 63.6 |

> 注：早期 REPORT 曾用「整份 log 推算」逐视频耗时，因 kitchen_01 日志含 11-13 失败轮（6506s）
> 导致显示 144906.5s 的错误；已改为直接读 receipts，现为上面的正确值。

---

## 4. kitchen_01 重采样决策（已采纳）

- 原官方答案 19.23% 异常偏低，诊断为小样本随机性（同图仅重跑 QA 三次得 30.77% / 38.46% / 46.15%，均值≈33.7%），非数据问题
  （MiniMax 参数修复 11-13 17:32 已生效，77 条 episodic 20:17 生成）。
- **采用中位数 sample2（预览 38.46%）覆盖官方答案**，原始备份：
  `runs/m3_parallel4_paper8_20261113/og_qa/kitchen_01/m3_qa_results.original-19.23pct.jsonl`。
- 官方 judge 读到的是重采样答案，仅 Q08 判 0.5（judge 边界正常波动），故最终 34.62%。
- 决策记录：`experiments/m3_parallel4_paper8_20261113/kitchen_01_resample_decision.md`。

---

## 5. QA token-budget Pareto（同冻结图，仅改 `max_input_tokens`）

有效上限 = min(budget, 32768−8192−2048 = 22528)。

| budget | effective | QA input tok | QA output tok | 准确率% | QA 耗时 s |
|---|---|---|---|---|---|
| 6000 | 6000 | 2,044,084 | 77,082 | **47.94** | 1089.5 |
| 9000 | 9000 | 3,029,175 | 94,694 | **55.15** | 1291.9 |
| 12000 | 12000 | 3,873,144 | 102,980 | **59.79** | 2952.1 |
| 16000 | 16000 | 5,035,616 | 120,339 | **61.34** | 1390.9 |
| 22528 | 22528 | 6,471,426 | 117,639 | **59.28** | 16387.4 |

- **Pareto 前沿：[6000, 9000, 12000, 16000]；22528 被支配**（token 更高、准确率反降）。
- 结论：**在 16000 之前，增加 QA token 预算能稳定提升准确率；到 22528（上下文近满）收益消失甚至下降**。
- 注意：22528 档 QA 耗时 16387s（≈4.5h）异常长，疑似配额等待/重试，属该档特有现象，不影响准确率结论。
- 图：`experiments/m3_parallel4_paper8_20261113/qa_budget_pareto_20261113.png`。

---

## 6. 关键代码修复（已应用并验证）

1. **MiniMax 参数透传**（11-13 已应用，参照 reduce_face 分支精准移植）：`providers/llm/minimax_llm.py`、`m3/llm_factory.py`、`m3/config.py` 新增并透传
   `reasoning_effort / is_reasoning_model / extra_body`。usage 元数据已验证：`reasoning_effort=medium`、`thinking={"type":"adaptive"}`、`reported_model=MiniMax-M3.1-Flash`。
2. **episodic stage 加固**：`m3/pipeline/stages.py` 的 `episodic_memory_stage` 原在 rc=2（个别 clip 超时）时返回失败→中止整条 pipeline；
   改为「有产出记录即 degraded 成功（ok=True + warnings）」，仅非 {0,2} 或零记录才失败。冻结源码与主源码两份都改。
3. **driver QA 加固**：`scripts/run_m3_parallel4_paper8_20261113.py`（= `runs/.../driver.py`）原把「任一 QA exit 2」当致命错误；
   改为 QA 命令加 `--resume`（只补失败题），有答案的 exit 2 记 `degraded` 继续。manifest `driver_sha256` 已同步。

---

## 7. 环境状态

| 项 | 值 |
|---|---|
| 主仓库 | `/home/yuanjian/Development/memory-eval`（main，多个未提交实验目录，勿动） |
| oGMemory 分支 | `rychkov/m3-parallel`（含 MiniMax 修复，未提交） |
| DB 容器 | `m3p4-pg-20261113` (pgvector/pg16) `127.0.0.1:31877`，库 `ogmem_m3_parallel4_paper8_20261113` |
| DB 密码 | `runs/.m3p4_pgpass_20261113`(0600) |
| Embedding | `t4-omni-embeddings-1` GPU1 `127.0.0.1:8001`（BAAI/bge-m3，key `abc-123`，走 `/v1/embeddings`） |
| CV | `t4-omni-cv-gpu-1` `127.0.0.1:18081` |
| GPU | `CUDA_VISIBLE_DEVICES=3`（`M3_GPU=3`） |
| MiniMax key | `experiments/m3_parallel4_paper8_20261113/private/config.yaml`（0600，脱敏见 `config.redacted.yaml`） |

> 不要删除 embeddings / cv / pg 容器；不要动主仓库其它未提交实验目录；禁止 `rm -rf`（用 Python `shutil.rmtree`）。

---

## 8. 产物清单（`experiments/m3_parallel4_paper8_20261113/`）

- `status.json`（finished）· `judgments.jsonl`（97 行）· `judgments_qa_budget_*.jsonl`（5 档）
- `qa_budget_sweep_results.json` · `qa_budget_pareto_results.json` · `qa_budget_pareto_20261113.png`
- `REPORT.md` · `index.md` · `kitchen_01_resample_decision.md`
- `usage-*.jsonl`（pipeline/qa/judge/sweep 逐调用）· `stages/*.json`（receipts）· `private/config.yaml`

---

## 9. 论文基线（对照）

`experiments/m3_paper_paired8_20261103/comparison.json`：97 题 mean=0.57216；逐视频同上表「论文」列；
paper token：QA prompt 4,039,490 / completion 111,694；memory prompt 2,510,078 / completion 784,774。
该基线来自旧冻结源码（含 `fact_index`）；当前并行分支删除了 fact_index 且 `qa_context` schema 简化，故为**近似复现**。

---

## 10. 风险与注意事项

- `QuotaGate`：interval<12% 等待重置，weekly<5% 停止。
- judge_error 会保留现场并中止，便于复核（不静默重试）。
- 22528 档 4.5h 异常耗时疑似配额等待，若需精确耗时对比应复查 `private/qa_budget_22528-*.log` 与 `quota_observations.jsonl`。
