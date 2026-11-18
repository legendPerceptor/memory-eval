# M3-Bench 全量并行实验 — 详细结果报告（full100-ds）

> 本报告面向 **2 页 PPT**，提供可直接引用的数据、图表与结论。所有数字均为本轮全量实测 / 可复算重建值。
> 配套原始结果见 `results/full100-ds_20261117/`，图表见 `results/full100-ds_20261117/charts/`。

---

## 0. 一句话结论

在 **4×Tesla T4** 上，用 **GPU 加速 ASR（faster-whisper）+ GPU Embedding（bge-m3/TEI）+ GPU CV（insightface）**，
以 **2 视频 worker × 4 stage worker** 的并行方式，完成 **100 视频 / 1276 题** 全量 M3-Bench：

- **总准确率 = 53.61%**（failed=0，judge_error=2）
- **单视频总处理时间中位数 705s（≈12min）**，纯计算合计约 **21.6h**（100 视频全阶段求和）
- **ASR 上 GPU 后提速 ~10.7×**（RTF 0.010 vs 0.105，受控微基准），是本次加速的核心收益点
- 成本结构（全量 100 视频口径）：**QA 52.7% + Episodic Memory 生成 38.9%** 为两大头（合计 ≈92%），ASR 仅 4.5%

> ⚠️ 口径说明：本轮为**降配版**（4 帧/30s clip、不送原生音频、bge-m3 embedding、deepseek-v4.1-flash-1017），
> 与论文 57.2% 基线（16 帧 + 送音频 + Qwen3-Emb + MiniMax）**仅近似可比，不能等同论文数字**。

---

## 1. 实验设置

### 1.1 数据集与规模
| 项目 | 数值 |
|---|---|
| 数据集 | M3-Bench robot split |
| 视频数 | 100 |
| 题目数 | 1276 |
| 完成度 | **100/100 视频，1276/1276 题（ALL COMPLETE）** |
| failed / judge_error | 0 / 2 |
| 收尾时间 | 2026-11-17 17:07:25 UTC（共 30 个 pass，前 27 轮因限流 / embedding 服务故障反复重试） |

### 1.2 硬件与算子分工（B 方向：ASR + Embedding 均上 GPU）
| GPU | 用途 | 服务/实现 |
|---|---|---|
| GPU0 | ASR（faster-whisper） | cuda / float16 |
| GPU3 | ASR（faster-whisper） | cuda / float16 |
| GPU1 | Embedding | TEI `BAAI/bge-m3` @ :8001（dim 1024） |
| GPU2 | CV | insightface @ :18081 |

- ASR 以 `--asr-gpus 0,3` 组成 GPU 池：每个并发视频子进程**独占一张卡**（写 `CUDA_VISIBLE_DEVICES`），结束归还。
- **并行度**：2 video workers（受 MaaS QA 限流约束）+ 4 stage workers（`OGMEM_M3_MAX_WORKERS=4`）。
- LLM：extraction / judge = `deepseek-v4.1-flash-1017`（modelarts-maas），QA 同族。

### 1.3 相对论文口径的配置差异（重要）
| 维度 | 本次 full100-ds（降配） | 论文版 8 视频 |
|---|---|---|
| 每 30s clip 帧数 | **4 帧** | 16 帧 |
| 原生音频送模型 | **否** | 是（audio_attached=True） |
| Episodic LLM | deepseek-v4.1-flash-1017 | MiniMax-M3.1-Flash |
| Embedding | bge-m3（本地 GPU TEI） | Qwen3-Embedding-0.6B |
| 切片 | 30s/clip | 30s/clip |

---

## 2. 加速效果

### 2.1 ASR：GPU vs CPU（核心收益，受控微基准）
| 设备 / 精度 | 实时率 RTF（越小越快） | 相对提速 |
|---|---|---|
| CPU `int8` | 0.105 | 1.0× |
| **GPU `cuda/float16`（T4）** | **0.010** | **~10.7×** |

> RTF = 处理耗时 / 音频时长。10.7× 把 ASR 从早期的主要瓶颈压到全流程占比仅 ~4.5%（见 §3）。

### 2.2 单视频与总耗时（重建口径，覆盖全部 100 视频）
| 指标 | 数值 | 说明 |
|---|---|---|
| 单视频总处理时间 中位数 | **705s（≈12min）** | 各阶段求和（ingest + QA） |
| 单视频 均值 | 778s | 同上 |
| 单视频 p25 / p75 | 602s / 899s | 分布偏右尾（长视频更慢） |
| 单视频 min / max | 328s / 2039s | |
| **纯计算合计（100 视频全阶段求和）** | **≈21.6h** | 非墙钟，未含并行折算与重启 |
| loop 墙钟总跨度（含重启 / 限流等待） | 28.5h | 8+ 轮 pass 反复重试，非纯计算 |

**说明**：本轮墙钟被**前 27 轮 pass 的限流 / embedding 服务故障重试**显著拉长（pass 1–27 长时间卡在 9/100），
因此墙钟（28.5h）不代表算法计算量；真正与算法相关的是**单视频处理时间（中位 705s）与纯计算合计（21.6h）**。
加速的主证据是 ASR 的 **10.7×**（§2.1），以及把 ASR 占比压到 4.5%。

![每视频处理耗时分布](results/full100-ds_20261117/charts/per_video_time_distribution.png)

---

## 3. 各阶段算子耗时占比（全量 100 视频口径）

> **数据来源与可信度**：全量逐阶段计时无法从 pipeline.log 直接取（除 9 个视频外，其余日志被后续 resume 轮覆盖或片段化）。
> 本报告改用**数据库节点创建时间戳重建**：
> - Episodic / 抽帧 / 人脸 / 音频：由 `graph_nodes.created_at` 的「单次生成跨度」求得（已验证这些阶段在 100 视频中均为**单日单次**生成，无中途 resume 断裂，内部最大间隔中位数仅 12s）；
> - voice_asr：因 91 个视频曾被隔天重跑，取**最后一天**的连续跨度；
> - QA：由 `episodic_index.jsonl → m3_qa_results.jsonl` 的**文件写入时间差**求得（在 9 个真实日志上误差 <5s，精确吻合）。
> 该方法在 9 个有完整日志的视频上与实测交叉校验：总 ingest 误差 −4%～−12%（小视频绝对量小、误差率大），可信。

### 3.1 阶段耗时与占比（每视频均值，合计 778s）
| 阶段 | 平均耗时 | 占比 |
|---|---:|---:|
| **QA（ReAct 问答，题目并发）** | 410.2s | **52.7%** |
| **Episodic Memory 生成** | 302.9s | **38.9%** |
| **ASR（voice_asr）** | 34.8s | **4.5%** |
| 预处理合计（切片+抽音频+抽帧+人脸+建节点） | 27.0s | 3.5% |
| 索引（build_index，取自 9 日志） | 2.8s | 0.4% |

细项均值（秒）：切片 3.6 / 抽音频 3.3 / 抽帧 9.9 / 人脸 9.8 / ASR 34.8 / Episodic 302.9 / 建节点 0.3 / QA 410.2 / 索引 2.8。

![阶段耗时占比](results/full100-ds_20261117/charts/stage_time_breakdown.png)

**解读**：
- **QA 与 Episodic Memory 合计 ≈92%**，是唯二主成本；ASR 因上 GPU 已降至 4.5%，不再是瓶颈。
- 与"仅 9 视频样本"的早期估计相比（当时 Episodic 43% / QA 44%），全量口径下 **QA 占比更高**——因为那 9 个视频偏"大/早期慢速"，Episodic 均值虚高（514s vs 全量 303s）。**全量口径更代表真实成本结构。**
- 每视频耗时主要随**视频时长（clip 数与对话轮数）**线性增长：短房间 ≈5min，长 kitchen/study ≈30min。

---

## 4. Token 消耗与 LLM 调用量

> ⚠️ **重要限制（如实声明）**：本轮 run **未落盘精确 token 用量**（无 usage 表 /文件、m3 pipeline 未调用 `set_usage_store`）。
> 下方 token 数为**基于「输出字符数 ÷ 4」的估算**，用于量级参考，**非精确计量**；输入 token 无法精确统计。

### 4.1 可测代理量（实测）
| 指标 | 数值 | 来源 |
|---|---:|---|
| LLM chat-completions 调用总数 | **8,595** | 100 份 pipeline.log 的 POST 统计 |
| ├ Episodic Memory LLM 调用 | 4,143 | 同上 |
| └ QA LLM 调用 | 4,452 | 同上 |
| Episodic 记录数（=clip 数，非重复） | 6,870 | episodic_memory.jsonl |
| Episodic 输出字符总数 | 3,602,529 | → 估算 **≈900,632 output tokens** |
| QA 唯一题目数 | 1,276 | 去重后（eval 文件含 2× 重复行共 2552） |
| QA 预测输出字符 | 653,976 | → 估算 **≈163,494 output tokens** |
| QA model_rounds 合计（去重） | 5,629 | ReAct 迭代轮次 |
| QA 检索执行次数（去重） | 6,269 | search_executions |
| ASR 转写片段数 | 24,653 | transcript_segments |

![LLM 调用量与 Token 估算](results/full100-ds_20261117/charts/llm_calls_and_tokens.png)

### 4.2 估算 token 汇总（仅供量级参考）
| 来源 | 输出 token 估算 |
|---|---:|
| Episodic Memory 生成 | ≈900,632 |
| QA 回答生成 | ≈163,494 |
| **合计（仅输出侧）** | **≈1,064,126** |

> PPT 建议：明确标注"token 用量未持久化，以上为基于调用次数 / 输出字符的**估算**"。

---

## 5. 按题型的准确率评分表

### 5.1 按推理维度聚合（题目含该维度即计入，可重叠）
| 推理维度 | 题数 | 准确率 | 评测 |
|---|---:|---:|---|
| Person Understanding | 548 | **65.4%** | ✅ 强项 |
| Multi-Hop Reasoning | 85 | 62.9% | ✅ 强项 |
| Multi-Detail Reasoning | 842 | 56.1% | ✅ 略高于均值 |
| Cross-Modal Reasoning | 476 | 53.4% | ⚠️ 略低于均值 |
| General Knowledge Extraction | 327 | **37.5%** | ❌ 明显短板 |

![按维度准确率](results/full100-ds_20261117/charts/accuracy_by_dimension.png)

### 5.2 按题目类型（top 14 by 题数）
![按题型准确率](results/full100-ds_20261117/charts/accuracy_by_question_type.png)

完整 22 条题型明细（按题数排序）：

| 题型 | 题数 | 准确率 |
|---|---:|---:|
| Multi-Detail（单） | 200 | 47.5% |
| Person Understanding, Multi-Detail | 189 | **72.0%** |
| General Knowledge（单） | 174 | 37.4% |
| Person Underst.+Cross-Modal+Multi-Detail | 145 | 62.8% |
| Cross-Modal+Multi-Detail | 130 | 50.0% |
| Person Understanding（单） | 121 | 62.4% |
| General Knowledge+Multi-Detail | 58 | 39.7% |
| Cross-Modal+General Knowledge+Multi-Detail | 49 | 31.6% |
| Cross-Modal（单） | 42 | 48.8% |
| Person Underst.+Cross-Modal | 42 | 64.3% |
| Cross-Modal+General Knowledge | 39 | 41.0% |
| Person Underst.+Multi-Hop+Multi-Detail | 27 | 57.4% |
| Person Underst.+Cross-Modal+Multi-Hop+Multi-Detail | 17 | 64.7% |
| Multi-Hop+Multi-Detail | 14 | **78.6%** |
| Multi-Hop（单） | 11 | 54.5% |
| Cross-Modal+Multi-Hop+Multi-Detail | 7 | **92.9%** |
| General Knowledge+Multi-Hop+Multi-Detail | 3 | 66.7% |
| Person Underst.+Cross-Modal+Multi-Hop | 2 | 25.0% |
| Person Underst.+Multi-Hop | 2 | 50.0% |
| Person Underst.+Cross-Modal+GenKnowledge+Multi-Detail | 2 | 50.0% |
| GenKnowledge+Multi-Hop（单） | 1 | 0.0% |
| Person Underst.+Cross-Modal+GenKnowledge+Multi-Hop+Multi-Detail | 1 | 0.0% |

**关键观察**：
- **Person Understanding 类**（含人物身份）普遍高（60%~72%），人物理解是强项；
- **General Knowledge Extraction**（通用知识抽取）是最大短板（单类仅 37.4%，与 Cross-Modal 组合更低至 31.6%）；
- 样本量小的组合题（n≤14）波动大，解读需谨慎。

---

## 6. 结论与 PPT 映射建议

### 6.1 核心结论
1. **可跑通全量**：100 视频 / 1276 题全量完成，准确率 **53.61%**，工程链路稳定（failed=0）。
2. **加速有效（核心是 ASR）**：ASR 上 GPU **10.7×**，占比降至 4.5%；单视频处理中位 **705s（12min）**，纯计算合计 ≈**21.6h**。
3. **成本结构**：**QA（52.7%）+ Episodic Memory（38.9%）≈92%** 为两大头，是后续优化的重中之重（二者都受 LLM 调用数驱动）；ASR/预处理已非瓶颈。
4. **能力画像**：人物理解强（65.4%），通用知识抽取弱（37.5%），跨模态中等（53.4%）；且本轮**未送原生音频**，多模态仍偏弱。

### 6.2 两页 PPT 内容映射
**第 1 页 —「做了什么 + 快不快」**
- 左：实验规模（100 视频 / 1276 题 / 53.61%）+ B 方向 4×T4 分工表
- 右上：加速效果（ASR 10.7×；单视频中位 705s；纯计算 21.6h）
- 右下：`per_video_time_distribution.png` 或 `stage_time_breakdown.png`

**第 2 页 —「效果如何 + 成本在哪」**
- 左上：`accuracy_by_dimension.png`（能力画像）
- 左下：按题型准确率表（§5.2 精选 top 类）
- 右：`llm_calls_and_tokens.png` + token 估算说明（标注"估算/未落盘"）+ 阶段占比结论（92% 在 QA+Episodic）

---

## 附：文件索引
| 文件 | 内容 |
|---|---|
| `results/full100-ds_20261117/README.md` | 基础版结果说明 |
| `results/full100-ds_20261117/m3_eval_summary.json` | 汇总（by_type / by_video / overall） |
| `results/full100-ds_20261117/m3_eval_results.jsonl` | 逐题评测明细（含重复行，按 question_id 去重后 1276 题） |
| `results/full100-ds_20261117/all_m3_qa_results.jsonl` | 原始 QA 结果 |
| `results/full100-ds_20261117/run_report.json` | 运行报告 |
| `results/full100-ds_20261117/charts/*.png` | 本报告 5 张图表 |
| `PLAN.md` | 加速方案与 10.7× 数据来源 |
| 原始数据 | `/data/m3_runs_yuanjian/full100-ds/`（阶段耗时由 `graph_nodes.created_at` + 文件 mtime 重建） |

> 图表均由实测/重建数据用 matplotlib 生成（英文标签，避免字体缺字）；正文表格已提供全部数值，可直接取用。
