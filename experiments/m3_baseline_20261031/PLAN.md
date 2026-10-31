# M3 五视频基线评测计划

记录日期：2026-10-31。此文件在首个视频生成答案之前建立。

## 目的与边界

评估当前 oGMemory `dev` 的完整多模态管线，建立以后改进的可追溯基线。
本轮不修改算法，不以历史实验的答案或记忆代替新运行。不开展 LoCoMo 评测。
结果只代表 M3 robot 子集的五视频开发性 pilot，不代表完整 M3-Bench。

## 固定设置

- 来源版本、配置哈希见 `provenance.json`，脱敏配置见 `config.redacted.yaml`。
- 入口：`../../scripts/run_m3_baseline.sh`，调用原仓库 `scripts/run_m3_random10.sh`。
- 从 100 个有题目的 robot 视频中排序后以 seed=42 无放回抽 5 个。
- `study_05` 12 题、`gym_03` 11 题、`bedroom_04` 11 题、
  `study_18` 14 题、`kitchen_20` 17 题；合计 65 题，不按结果换样本。
- `study_05` 已见于历史开发实验，本样本不是独立留出集。
- 完整视频、30 秒片段、1 fps、至多 16 帧、512 像素缩放、JPEG quality 80；
  音频为 16 kHz 单声道，原始音频和 ASR 均启用。
- 情景记忆 / QA / judge：在线 `omni-7b`（Qwen2.5-Omni-7B）。
  原默认配置写 `omni-3b`，与实时 `/v1/models` 不一致，只纠正名称。
- 记忆输出上限 1024、temperature=0.2；QA 输出上限 4096、temperature=0；
  judge 输出上限 4096、temperature=0；请求超时继承 300 秒。
- BGE-M3 1024 维；faster-whisper large-v3-turbo（英语、beam=5）；
  pyannote speaker-diarization-3.1；SpeechBrain ECAPA。
- QA 至多 6 轮、循环预算 180 秒，默认 top-k=8、candidate-k=20、
  transcript-top-k=20、RRF 融合、context-k=8、邻居半径 1。
- `current_clip` 时间策略；允许问题所在片段结束前的上下文，
  **不是严格因果评测**。实际未来宽限由每题 `temporal_scope` 审计。
- `qa_context` 未启用，保持 wrapper 默认；本次不混入有界上下文改进。
- 专用数据库 `ogmem_m3_baseline_20261031`、账号 `m3-baseline-20261031`、
  向量表 `m3_baseline_20261031`。新库启用 vector 扩展，不复制旧数据。
- 原始输出置于 `../../runs/m3_baseline_20261031/`，凭据和未经审查日志不入 Git。

## 统计契约

1. 以 manifest 的 65 题为固定分母，关联 `(video_id, question_id)`。
2. 区分视频执行成功、记忆覆盖、QA 记录覆盖、非空无错误答案、有效评分。
3. 自动评分为 `{0, 0.5, 1}`。报告题目加权均分与视频宏平均，
   不称为独立人工准确率；同模型作答和评分存在相关偏差。
4. 缺失答案与执行失败在端到端效用分中计 0，并单列数量。
   judge 故障不得解释为模型回答错误；如出现，主质量均分标为不可用，
   另给明确注明故障计零的保守操作性指标。
5. 保留首轮失败与恢复记录。不从多次答案中择优，不静默更换模型/预算。
6. 单视频成功不代表全部阶段都成功；检查 summary、失败文件及非空向量。
7. 若配置变更或补充诊断，使用单独名称/输出，不混进原始基线。

## 已知限制与待核实风险

- 16 帧来自每片段前约 0–15 秒，并非整段均匀覆盖。
- 1024 的记忆输出预算可能导致截断；应如实保留失败，不把残缺 JSON 当成功。
- 32K 在线上下文与未启用的有界 QA 之间存在容量风险。
- 完整管线生成身份关系与语义汇总节点，但默认 QA 的主检索对象是
  `episodic_memory_generator` ExtractedNode；不能把执行全部阶段等同于
  已证实身份图/语义记忆对问答有增益。
- 只测一个配置、五个视频，不能证明因果增益或推导可靠总体置信区间。

## 环境事件

- 17:37 启动；新库缺少 vector 扩展导致 `study_05`、`gym_03`
  在媒体处理前初始化失败。原日志保存于 `private/setup-failures/`。
- 已启用 vector 0.8.7；后续成功运行同一固定样本，不通过换视频规避故障。
- M3 单元测试实际运行：759 passed in 11.43s。
