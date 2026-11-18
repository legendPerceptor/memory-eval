# 8 视频算子消融实验计划（episodic + ASR 最小臂 及 人脸/说话人贡献验证）

创建时间：2026-11-16（Asia/Shanghai）
作者：Codex（基于对代码与历史实验的审计）

本文件是**设计文档**，尚未启动任何新实验。任何"要跑"的行为都在下文"执行清单"里明确列出，
需要用户确认后再跑（每臂约 6 小时 pipeline + 20 分钟 QA）。

---

## 0. 一句话结论（先看这个）

1. 你直接要的"只保留 episodic memory + ASR、其余全关"实验**可以做**，命名 `m3_ablation8_episodic_asr_20261116`，
   预期准确率会**明显低于**当前 63.92% 基线——但差异**不能**归因于"人脸被关掉"，因为当前分支本身就没用人脸。
2. **`dev` / `m3-parallel` 里根本没有 person_grounding**（那是论文期未合入的实验分支 `m3_dev_1031`/`8731930` 才有）。
   所以在当前主分支上做"关人脸"消融是**结构性 null**（人脸本就没接进 QA），别指望它给人脸定罪或平反。
   真要测人脸贡献，得在论文实验分支上做 `FULL vs FULL-人脸`，或把该能力作为新特性移植进 dev 后再消融（见 `CAPABILITY_GAP.md`）。
3. speaker diarization 的贡献是**低/待测**；speaker_embedding / voice_clustering / voice_segmentation /
   语义整合 / face_clustering 在当前链路里**对 QA 的直接贡献为 0**（下游没有消费者）。
4. 建议**先跑 2–3 个高信息量臂**（最小臂 + 论文源码 FULL + 论文源码 FULL-人脸），而不是一次性跑 10 个组合。

---

## 1. 目标与要回答的问题

用户核心诉求：

- **Q1（直接要求）**：只开 episodic memory + ASR 语音转写，把其余算子全关，8 视频 97 题的准确率是多少？
- **Q2（成本质疑）**：在人脸识别和 speaker diarization 上花了不少成本，它们**真的都有贡献**吗？
- **Q3（方法学）**：整理所有可选算子，判断它们是否真的对 QA 有贡献，并据此设计不同算子的消融实验。

---

## 2. 固定实验样本（所有臂共用）

| 项目 | 值 |
|---|---|
| 视频（8） | `study_09, kitchen_01, bedroom_05, study_23, kitchen_23, study_05, gym_03, bedroom_04` |
| 题目 | `/data/m3_data/annotations/robot.json`，共 **97 题** |
| 每题数 | 12/13/12/14/12/12/11/11 |
| 模型 | `MiniMax-M3.1-Flash`（thinking=adaptive, reasoning_effort=medium） |
| 数据库 | 每臂一个全新 pgvector 库 `ogmem_m3_ablation8_*`，禁止复用/覆盖已有库 |
| 评分 | 0 / 0.5 / 1 均值；缺失或失败按 0；同一 judge 模型；judge_seed 固定 |

题目能力标签分布（同一题可多标签）：Multi-Detail Reasoning 58、General Knowledge Extraction 38、
Person Understanding 33、Cross-Modal Reasoning 32、Multi-Hop Reasoning 5。

**当前分支近似基线**：`experiments/m3_parallel4_paper8_20261113` = **63.92%**（97 题，fresh DB，
MiniMax-M3.1-Flash）。注意这是"近似复现"，不是论文 57.2% 的字节级复现。

---

## 3. 算子清单与"是否真的对 QA 有贡献"判断

> 判断依据：QA 检索侧只消费 `extractor_type = "episodic_memory_generator"` 节点的文本
> （`m3/search.py` 第 ~264/333/392 行、`m3/control_adapter.py` 默认 `extractor_types=("episodic_memory_generator",)`）。
> 检索文本来自 `m3/index_episodic.py`：description / characters / events / entities / speech_transcript(ASR)。
> `same_identity` 边、`MemoryNode`、`voice_segmentation` 结果**都没有进入 QA 检索**。

| 阶段 id | 类型 | 角色 | 对 QA 的直接贡献 | 说明 / 证据 |
|---|---|---|---|---|
| trim / annotate / preprocess / load_clips | m3.* | 基础设施 | 无（链路必需） | 切 clip、写标注，无算子语义 |
| extract_frames | m3.extract_frames | 必要输入 | 间接（核心来源） | episodic 视觉记忆的帧来源 |
| extract_audio | m3.extract_audio | ASR 原料 | 无（原料） | voice_asr 的前置，本身无语义 |
| **voice_asr** | m3.voice_asr | **核心** | **高** | ASR transcript 写入 graph → `m3/episodic_memory.py::_index_asr_segments` 进 prompt；同时写 `speech_transcript`，`index_episodic` 置顶到检索文本，是 QA evidence |
| **speaker_diarization** | m3.speaker_diarization | 说话人匿名标注 | **低 / 待测** | 只给 ASR segment 回写匿名 `speaker_id`，episodic prompt 显示 `SPEAKER_00: text`。匿名标签对"谁在说话/对话归属"可能有边际帮助 |
| speaker_embedding | m3.speaker_embedding | 向量 | **0（当前链路）** | 只供 voice_clustering 用，不进 prompt / 索引 / QA |
| voice_segmentation | m3.voice_segmentation | 能量分段 | **0（当前链路）** | 仅供 voice_clustering；无 QA 消费者（voice_clustering 的输出也不被 QA 用） |
| voice_clustering | m3.voice_clustering | same_identity 边 | **0（当前链路）** | 生成 `same_identity` 边，QA 检索不使用 |
| **detect_faces** | m3.detect_faces | 人脸检测 | **0（dev/parallel 链路）** | dev 的 episodic `needs` 不含人脸、无 person_grounding；仅论文实验分支 `m3_dev_1031` 才接入 prompt（见 §4） |
| face_clustering | m3.face_clustering | same_identity/人物候选 | **0（dev/parallel 链路）** *(仅论文实验分支下经 person_grounding 间接生效)* | dev 无 causal_identities/person_identities 接线 |
| **episodic_memory** | m3.episodic_memory | **核心** | **核心** | 生成可检索的记忆节点（description/characters/events/entities/transcript） |
| create_memory_nodes | m3.memory.from_episodic | 记忆节点 | 基础设施/可能 0 | QA 检索的是 episodic extracted node；MemoryNode 未直接进检索 |
| **build_index** | m3.index_episodic | **必需** | **必需** | QA 检索依赖向量索引 |
| semantic_memory_consolidator | m3.semantic_memory_consolidator | 语义整合 | **0（当前链路）** | 生成 MemoryNode，但未进入 index_episodic / QA 检索 |
| qa / judge | m3.qa / eval | 评测 | 评测 | 非记忆算子 |

**净结论（当前分支 `rychkov/m3-parallel`）**：
- 有实质 QA 贡献的算子只有：`voice_asr`（高）、`episodic_memory`（核心）、`build_index`（必需），
  加上必要输入 `extract_frames` / `extract_audio`。
- `speaker_diarization` 低/待测；`speaker_embedding` / `voice_segmentation` / `voice_clustering` /
  `face_clustering` / `semantic_memory_consolidator` 在当前链路上对 QA **直接贡献为 0**。
- `detect_faces` 的贡献**只取决于源码版本**（见 §4）。

---

## 4. ⚠️ 关键更正（2026-11-16）：person_grounding 从未进 dev；parallel 没删东西

> **修正**：早先版本误把"论文冻结快照 `8731930`"当作基准，得出"m3-parallel 删了 person_memory / fact_index"的错误结论。
> 经严格核对（见 `CAPABILITY_GAP.md`），正确事实如下：

- **`dev` 是 `rychkov/m3-parallel` 的祖先**（`0 156`，parallel = dev + 156 提交），**没有任何文件/函数被 parallel 删掉**。
- **`person_memory.py` / `person_grounding` / `fact_index.py` / `fact_memory.py` 都不在 `dev`，也不在 `master`**；
  它们只存在于论文期的实验分支 `m3_dev_1031` / `simplify_person_info_1110` / `intergation_1103`（含提交 `8731930`），
  **从未合入主干**。所以不是 parallel 删除，而是"本就未进 dev"。
- **`detect_faces` / `face_clustering` 在 dev 里照常存在**，但 `episodic_memory` 只 `needs: [extract_frames, extract_audio, voice_asr]`，
  dev 的 pipeline **没有** `person_identities` / `person_grounding` 接线；`face_clustering` 也**没有** `causal_identities`。
- **`same_identity` 边 / `MemoryNode` 不进 QA 检索，是 dev 的既有设计**（dev 里消费者同样只有聚类自身 + 清理）。

**因此（修正后的实验结论）：**
- 在 `m3-parallel`（≈ dev + 性能）上跑"关人脸 / 关 diarization"消融 → 会得到**诚实的近似 null**；
  这是"当前主分支设计下这些算子对 QA 无直接贡献"的证据，**不是** parallel 的缺陷或回归。
- 要验证"人脸识别成本是否值得"，必须在**含 person_grounding 的论文分支（`m3_dev_1031` / `8731930`）**上做
  `FULL vs FULL-人脸`；或先把该能力作为**新特性**移植进 dev 并接好 QA 检索，再在同一基线消融。

---

## 5. 实验臂设计

### A 组 —— 当前分支 `rychkov/m3-parallel`（回答 Q1：最小记忆 + ASR）

**A1 臂 `m3_ablation8_episodic_asr_20261116`（用户直接要求，最高优先级）**

- 开启：`extract_frames`, `extract_audio`, `voice_asr`, `episodic_memory`, `create_memory_nodes`, `build_index` + 独立 QA
- 关闭：`detect_faces`, `face_clustering`, `speaker_diarization`, `speaker_embedding`,
  `voice_segmentation`, `voice_clustering`, `semantic_memory_consolidator`
- `episodic_memory.params.no_audio=true`（只保留 ASR 文本，避免原始音频混入"ASR-only"条件）
- QA 参数与 `m3_parallel4_paper8_20261113` 完全一致：
  `--top-k 8 --candidate-k 20 --transcript-top-k 20 --transcript-fusion rrf --context-k 8
   --include-neighbors 1 --max-iterations 6 --timeout-seconds 180 --temporal-policy current_clip`
- 全新 DB `ogmem_m3_ablation8_episodic_asr_20261116`，拒绝覆盖
- **预期**：明显低于 63.92%（丢了人物/Speaker/Voice/语义整合等上下文）。
  **但不把差异归因于人脸**——当前分支 full 基线本身也没用 face prompt。

### B 组 —— 论文冻结源码 `8731930`（回答 Q2：人脸是否真的贡献）

> 复用 `experiments/m3_paper8_fresh_20261110` 的**冻结源码快照 + 配置 schema + 独立 QA/Judge 参数**，
> 只改算子开关。三臂必须在**同一份冻结源码、同模型、同 QA/Judge 参数、同全新 DB 策略**下对照。

**B0 `FULL`（论文完整 pipeline 基线）**
- 若 `experiments/m3_paper8_fresh_20261110` 已有完整最终评分（实测：**有**，fresh=50.52%，
  paper_og=55.15%，paper_m3=44.85%，97 题全完成），且源码/配置/样本一致，**优先直接复用**为 FULL 对照，
  省一臂 ~6 小时。前提是核对 manifest 的 `source.commit=8731930`、`person_grounding=True`、
  8 视频 97 题、QA/judge 参数与本次一致。
- 若需与 A 组用同一 judge seed / 同一模型窗口，则重跑一臂以保证可比。

**B1 `FULL − face/person_grounding`（人脸贡献的关键对照）**
- `detect_faces.enabled=false`
- `face_clustering.enabled=false`（同时也就不生成 `person_identities`）
- `episodic_memory.params.person_grounding=false`
- 其余算子（含 voice_asr / speaker_diarization / speaker_embedding / voice_* / 语义整合）保持与 FULL 一致
- 全新 DB `ogmem_m3_ablation8_noface_20261116`
- **`B0 − B1` 的差 = 人脸识别（+人物裁剪接地）对 QA 的真实贡献**。

### C 组 —— 其余算子的逐项消融（回答 Q3，按性价比排序）

均基于论文源码 FULL（保证有下游消费者可测），逐项关一个、其余不动：

| 优先级 | 臂 | 改动 | 预期信息量 |
|---|---|---|---|
| 高 | `FULL − speaker_diarization` | 关 `speaker_diarization`，保留 voice_asr（transcript 仍在，只是无 speaker_id） | diarization 是否值得（用户成本质疑的另一半） |
| 中 | `FULL − semantic_memory_consolidator` | 关语义整合 | 验证"语义整合 0 贡献"假设 |
| 低/合并 | `FULL − 聚类算子`（一次性关 `face_clustering`+`voice_clustering`+`speaker_embedding`+`voice_segmentation`） | 预期 null | 一次确认"聚类类贡献≈0"，省成本 |
| 可选 | `FULL − voice_asr` | 关 ASR，只留视觉 episodic | 验证 ASR 高贡献（预期大跌） |

> 不建议一开始就跑 10 个组合。建议顺序：**A1 → B0(尽量复用) → B1 → C(speaker_diarization) → C(semantic) → 视结果再决定聚类/voice_asr**。

---

## 6. 驱动设计（需要新写的代码）

基于 `scripts/run_m3_parallel4_paper8_20261113.py` 的结构（`prepare / worker phase video / judge / run`），
新增 `scripts/run_m3_ablation8_20261116.py`，关键要求：

1. **臂参数化**：每臂独立 `NAME / ACCOUNT / DB / pipeline 开关`，从一份 `ARMS` 字典驱动，避免复制粘贴出错。
2. **显式 stage 开关**：worker 里向 `run_video_pipeline` 传
   `--disable-stage detect_faces --disable-stage face_clustering --disable-stage speaker_diarization
    --disable-stage speaker_embedding --disable-stage voice_segmentation --disable-stage voice_clustering
    --disable-stage semantic_memory_consolidator`（A1 臂）；
   再 `--set episodic_memory.params.no_audio=true`。
   （`--disable-stage` 由 `PipelineRunner.apply_overrides` 处理，禁用阶段记为 `skipped`，
   下游 `needs` 检查对"依赖已 skipped 但 ok=True"是放行的——已核对 `runner.py` 逻辑。）
3. **源码版本选择**：
   - A 组：`git archive` 冻结当前 `rychkov/m3-parallel` HEAD + 工作区 `source.patch`（沿用 parallel4 驱动做法）。
   - B/C 组：直接 `copytree` 论文冻结快照 `runs/m3_paper8_fresh_20261110/_source/oGMemory`，
     并冻结 `person_grounding=true` 的 pipeline yaml 副本（`--set` 关闭项）。
4. **全新 DB + 拒绝覆盖**：`psycopg2` 建库（沿用 `run_m3_ten_video.py::prepare` 的 `CREATE DATABASE` 逻辑），
   先查 `pg_database` 是否存在，存在则拒绝；`RUN/EXP` 已存在则 `FileExistsError`。
5. **可恢复**：`stages/<video>-<phase>.json` 收据 + QA `--resume`；失败阶段需人工 review，不自动覆盖。
6. **预检模式**：`--selection-only` / `--preflight`，只校验 8 视频、97 题、DB 可建、源码 hash、
   模型配置，**不调用模型**。
7. **manifest 记录**：source_commit / 分支 / patch 或 snapshot hash、每臂 stage 开关、
   `person_grounding` 值、QA/judge 参数、DB 名、模型配置、`safety_stop_at_unix`。
8. **环境**：`CUDA_VISIBLE_DEVICES` 走 `M3_GPU`（默认 GPU3，当前空闲）；
   `MINIMAX_API_KEY` 从 `experiments/m3_reduce_face_qa_budget_20261113/private/config.yaml` 恢复
   （当前 shell 未导出）；`HUGGINGFACE_TOKEN/HF_TOKEN` 已设（diarization 需要）。

---

## 7. 评分与可比性协议

- 每臂独立 QA 输出 → 独立 judge；也可把多臂答案**一起盲评**（同一 judge，随机顺序，固定 seed）以减少
  judge 漂移，参考 `m3_paper8_fresh_20261110` 的 joint 盲评做法。
- 缺失/失败答案一律 0 分，保留原始输出与错误，不选择性重跑。
- **可比性三要素必须一致**：同 97 题、同模型与参数、同 judge；DB 全新；源码版本在组内一致（A 组内部一致；
  B/C 组内部一致）。**跨源码版本（A vs B）不能做因果归因**，仅作诊断参考。
- 报告需分视频给出 mean score，并标注每题标签，便于看 Person Understanding 子集变化。

---

## 8. 成本与停止条件

- 单臂成本：约 **6 小时 pipeline（8 视频串行，stage 并发 4）+ ~20 分钟 QA**（含额度等待可能更久）。
- GPU：4×T4 15GB，当前 0/1/2 有占用、GPU3 空闲 → 单臂串行即可，不必并行抢卡。
- 停止条件（写入驱动）：
  - `safety_stop_at_unix = 启动 + 12h`（单臂）超时即停；
  - 预算/额度门控（沿用 `QuotaGate`）；
  - 某阶段 `exit_code != 0` 且非可降级的 QA 部分失败 → 标记 `needs_review` 停；
  - DB 已存在、源码 hash 不匹配、模型配置校验失败 → 拒绝启动。

---

## 9. 执行清单（需要用户确认后才跑）

**Phase 0 — 预检（不花钱，几分钟）**
- [ ] 写 `scripts/run_m3_ablation8_20261116.py`（含 `--preflight`）。
- [ ] `--preflight` 验证：8 视频存在、97 题、A1 DB 可建、source commit/patch hash、模型配置、API key 可恢复。
- [ ] 核对 `m3_paper8_fresh_20261110` 能否直接复用为 B0 FULL（source.commit/person_grounding/QA 参数/样本一致）。

**Phase 1 — A1 最小臂（用户直接要求）**
- [ ] `prepare` → 启动 `m3_ablation8_episodic_asr_20261116`（全新 DB）。
- [ ] pipeline → QA → judge；产出 REPORT。

**Phase 2 — B0/B1 人脸贡献**
- [ ] 有可复用 FULL 则直接做 B1（`FULL−face`），否则先补 B0。
- [ ] `B0 − B1` 得人脸贡献结论。

**Phase 3 — C 组按结果决定**（speaker_diarization → semantic → 聚类/voice_asr）

---

## 10. 风险与注意事项

1. **不要在当前分支做"关人脸"**：结构性 null（见 §4）。
2. **A1 的准确率下降不等于"人脸有用"**：它是"丢了一堆上下文"的综合结果，且当前 full 基线本就没人脸。
3. **跨源码版本不可因果比较**：A 组（`rychkov/m3-parallel`）与 B/C 组（`8731930`）代码不同，
   只能各自组内对照。
4. **工作区有未提交修改**（`m3/config.py`、`m3/llm_factory.py`、`m3/pipeline/stages.py`、
   `providers/llm/minimax_llm.py`），冻结源码时用 `git archive` + `source.patch`，**不要覆盖**这些改动。
5. **diarization 需要 HF token + EULA**：pyannote 模型需在 HF 接受协议；无 token 会运行期报错。
   A1 臂已关闭 diarization，不受影响。
6. **重复成本**：一臂 6 小时，务必先 Preflight + 复用 B0，避免重复跑基线。

---

## 11. 关键参考路径

- 当前完整管线：`oGMemory/config/m3/pipelines/video_episodic_full.yaml`
- 阶段实现：`oGMemory/m3/pipeline/stages.py`；runner 覆盖/禁用：`oGMemory/m3/pipeline/runner.py`
- QA 检索：`oGMemory/m3/search.py`、`oGMemory/m3/control_adapter.py`
- ASR 进记忆/索引：`oGMemory/m3/episodic_memory.py`、`oGMemory/m3/index_episodic.py`
- 现有 8 视频驱动：`scripts/run_m3_parallel4_paper8_20261113.py`
- 现有 8 视频结果（63.92%）：`experiments/m3_parallel4_paper8_20261113/REPORT.md`
- 论文冻结源码（含 person_grounding）：`runs/m3_paper8_fresh_20261110/_source/oGMemory/`
  - `m3/person_memory.py`、`m3/pipeline/stages.py`(~408)、`config/m3/pipelines/video_episodic_full.yaml`
- 论文冻结最终评分：`experiments/m3_paper8_fresh_20261110/{comparison.json,REPORT.md,manifest.json}`

---

## 12. 追加核对：B0 复用 paper8_fresh 的一致性要求（2026-11-16）

> 注：paper8_fresh 用的是**论文实验分支**（`m3_dev_1031`/`8731930`，含 person_grounding / fact_index），
> 与 dev / m3-parallel 不是同一代码基线。B 组只用于回答"论文那套人脸/身份能力贡献"，不代表当前主干。

实测 `experiments/m3_paper8_fresh_20261110/manifest.json` 与 parallel4 的差异，做 B1 时必须**照抄 paper8_fresh 的设置**，
而不是照抄 parallel4，否则 B0−B1 不可比：

- **qa_context（paper8_fresh 更丰富，B1 必须一致）**：
  `fact_memory=true, identity_aware=true, incremental_delivery=true, evidence_format=grouped,
   force_answer_tool=true, answer_reserve_seconds=45, incremental_tokens=3500, max_input_tokens=24000` 等
  （parallel4 的 qa_context 没有这些字段——**不能混用**）。
- **模型/评分**：`model=MiniMax-M3.1-Flash, thinking=adaptive, reasoning_effort=medium,
  qa_output_tokens=8192, judge_output_tokens=2048, max_iterations=6, timeout_seconds=180, temporal_policy=current_clip`（与 parallel4 相同）。
- **样本**：`selected_video_ids` 8 个、97 题（一致）。
- **source**：`commit=8731930`、`person_grounding=true`（一致）。
- **B0 数值**：paper8_fresh 的 `ogmemory_strict_incremental`（fresh）= **50.52%**（3 题失败按 0）。
  这是 B 组的 FULL 参照，**不是** 63.92%（那是当前分支 parallel4）。
- **结论**：B1 需在同一冻结源码 + 同一 paper8_fresh 版 qa_context/QA 参数下只关人脸/person_grounding，
  才能用 `50.52% − B1` 解释人脸贡献。
