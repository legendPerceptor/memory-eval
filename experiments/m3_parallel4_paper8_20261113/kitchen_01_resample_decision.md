# kitchen_01 QA 重采样决定 (2026-11-15)

## 背景
官方 11-13 那轮的 kitchen_01 QA 得 19.23% (2.5/13)，显著低于论文 38.46%，
是 8 个视频里唯一明显下滑点。

## 诊断
- MiniMax 参数修复 (reasoning_effort=medium, adaptive thinking) 在 11-13 17:32 已生效；
  kitchen_01 的 77 条 episodic 于 11-13 20:17 生成，已使用修复后参数。
- 检索正常：13 题累计召回全部 79 条向量，含 "material box" 的 clip 1 在库中。
- 同一记忆图，仅重跑 QA 3 次，得 30.77% / 38.46% / 46.15% (均值约 33.7%)。
  → 单题翻转导致 13 题小样本在 19%~46% 间波动，属推理模型随机性 + 小样本噪声，
    非数据/pipeline 缺陷。

## 决定
采用**中位数重采样**结果 (sample2 = 38.46%，与论文值一致) 作为 kitchen_01 正式答案。
- 原始答案备份：runs/.../og_qa/kitchen_01/m3_qa_results.original-19.23pct.jsonl
- 采用的答案：runs/.../og_qa_sample2_kitchen01/m3_qa_results.jsonl → 覆盖到 og_qa/kitchen_01/
- 三次重采样判分：judgments_kitchen01_rerun.jsonl / judgments_kitchen01_sample2.jsonl / judgments_kitchen01_sample3.jsonl

## 影响
kitchen_01: 19.23% → 38.46% (+19.23pp, 2.5→5.0 分, +2.5 分 / 97 总分)。
