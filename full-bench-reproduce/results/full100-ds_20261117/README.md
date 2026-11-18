# M3-Bench 全量复现结果 — full100-ds（2026-11-17 跑完）

本目录是 `full-bench-reproduce` 全量 M3-Bench 实验的**最终结果与报告**，供论文写作 / PPT 制作 / 其他 agent 直接读取。原始数据仍在 `/data/m3_runs_yuanjian/full100-ds/`，这里只放汇总与合并结果。

## 实验概况

- **数据集**: M3-Bench robot split
- **视频数**: 100
- **题目数**: 1276
- **已答/失败/judge_error**: 1276 / 0 / 2
- **总准确率 overall_mean_score**: 0.5361  (53.61%)
- **judge_provider**: modelarts-maas
- **judge_model**: deepseek-v4.1-flash-1017

## 实验配置（重要：这是“降配版”，非严格论文口径）

| 维度 | 本次全量 full100-ds | 论文版 8 视频 |
|---|---|---|
| 抽帧 vs 送视频 | 抽帧，`video_input=False` | 抽帧，`video_input=False` |
| 每 30s clip 帧数 | **4 帧** (`episodic_memory.params.max_frames=4`) | **16 帧** (`--episodic-max-frames 16`) |
| 原生音频送模型 | **否** (`no_audio=true`) | **是** (`audio_attached=True`) |
| ASR 文本 transcript | 有 | 有 |
| Episodic LLM | deepseek-v4.1-flash-1017 | MiniMax-M3.1-Flash |
| Embedding | bge-m3（TEI，本地 GPU） | Qwen3-Embedding-0.6B |
| 切片长度 | 30s/clip | 30s/clip |

> 注意：因帧数更少(4 vs 16)、且未送原生音频、embedding 与 LLM 模型也不同，本结果与论文 57.2% 基线**仅近似可比，不能直接等同论文数字**。

## 分题型准确率（by_type）

| 题型 | 准确率 | 题数 |
|---|---|---|
| Cross-Modal Reasoning,Multi-Hop Reasoning,Multi-Detail Reasoning | 92.9% | 7 |
| Multi-Hop Reasoning,Multi-Detail Reasoning | 78.6% | 14 |
| Person Understanding,Multi-Detail Reasoning | 72.0% | 189 |
| General Knowledge Extraction,Multi-Hop Reasoning,Multi-Detail Reasoning | 66.7% | 3 |
| Person Understanding,Cross-Modal Reasoning,Multi-Hop Reasoning,Multi-Detail Reasoning | 64.7% | 17 |
| Person Understanding,Cross-Modal Reasoning | 64.3% | 42 |
| Person Understanding,Cross-Modal Reasoning,Multi-Detail Reasoning | 62.8% | 145 |
| Person Understanding | 62.4% | 121 |
| Person Understanding,Multi-Hop Reasoning,Multi-Detail Reasoning | 57.4% | 27 |
| Multi-Hop Reasoning | 54.5% | 11 |
| Cross-Modal Reasoning,Multi-Detail Reasoning | 50.0% | 130 |
| Person Understanding,Multi-Hop Reasoning | 50.0% | 2 |
| Person Understanding,Cross-Modal Reasoning,General Knowledge Extraction,Multi-Detail Reasoning | 50.0% | 2 |
| Cross-Modal Reasoning | 48.8% | 42 |
| Multi-Detail Reasoning | 47.5% | 200 |
| Cross-Modal Reasoning,General Knowledge Extraction | 41.0% | 39 |
| General Knowledge Extraction,Multi-Detail Reasoning | 39.7% | 58 |
| General Knowledge Extraction | 37.4% | 174 |
| Cross-Modal Reasoning,General Knowledge Extraction,Multi-Detail Reasoning | 31.6% | 49 |
| Person Understanding,Cross-Modal Reasoning,Multi-Hop Reasoning | 25.0% | 2 |
| Person Understanding,Cross-Modal Reasoning,General Knowledge Extraction,Multi-Hop Reasoning,Multi-Detail Reasoning | 0.0% | 1 |
| General Knowledge Extraction,Multi-Hop Reasoning | 0.0% | 1 |

## 逐视频准确率（by_video）

| 视频 | 准确率 | 题数 |
|---|---|---|
| kitchen_02 | 100.0% | 11 |
| bedroom_08 | 96.2% | 13 |
| study_03 | 86.7% | 15 |
| office_08 | 86.4% | 11 |
| study_20 | 84.6% | 13 |
| living_room_22 | 84.6% | 13 |
| study_15 | 81.8% | 11 |
| bedroom_04 | 81.8% | 11 |
| living_room_11 | 80.8% | 13 |
| study_02 | 80.0% | 10 |
| living_room_07 | 79.2% | 12 |
| kitchen_04 | 79.2% | 12 |
| living_room_18 | 76.9% | 13 |
| living_room_09 | 76.9% | 13 |
| living_room_16 | 73.1% | 13 |
| living_room_03 | 73.1% | 13 |
| living_room_08 | 71.4% | 14 |
| meeting_room_06 | 70.8% | 12 |
| kitchen_17 | 70.8% | 12 |
| study_11 | 70.0% | 10 |
| bedroom_01 | 70.0% | 15 |
| kitchen_13 | 69.2% | 13 |
| kitchen_08 | 66.7% | 9 |
| bedroom_07 | 66.7% | 15 |
| kitchen_21 | 65.0% | 10 |
| study_08 | 64.3% | 14 |
| kitchen_03 | 64.3% | 14 |
| meeting_room_02 | 63.6% | 11 |
| living_room_17 | 63.3% | 15 |
| study_01 | 62.5% | 12 |
| office_01 | 62.5% | 12 |
| living_room_06 | 62.5% | 16 |
| bedroom_02 | 62.5% | 12 |
| kitchen_10 | 61.5% | 13 |
| meeting_room_01 | 61.1% | 9 |
| gym_03 | 59.1% | 11 |
| bedroom_11 | 58.3% | 12 |
| bedroom_10 | 58.3% | 12 |
| bedroom_09 | 58.3% | 12 |
| study_13 | 57.7% | 13 |
| office_05 | 57.7% | 13 |
| living_room_01 | 57.7% | 13 |
| kitchen_14 | 57.7% | 13 |
| gym_04 | 57.7% | 13 |
| study_23 | 57.1% | 14 |
| kitchen_11 | 56.7% | 15 |
| kitchen_20 | 55.9% | 17 |
| living_room_12 | 54.2% | 12 |
| kitchen_19 | 53.8% | 13 |
| study_21 | 53.6% | 14 |
| study_12 | 53.6% | 14 |
| study_10 | 53.6% | 14 |
| bedroom_06 | 53.6% | 14 |
| study_16 | 50.0% | 12 |
| study_14 | 50.0% | 12 |
| study_07 | 50.0% | 13 |
| office_03 | 50.0% | 12 |
| living_room_13 | 50.0% | 14 |
| living_room_05 | 50.0% | 15 |
| kitchen_23 | 50.0% | 12 |
| kitchen_12 | 50.0% | 20 |
| kitchen_06 | 50.0% | 13 |
| office_04 | 46.4% | 14 |
| kitchen_16 | 46.4% | 14 |
| study_04 | 45.8% | 12 |
| office_02 | 45.8% | 12 |
| bedroom_05 | 45.8% | 12 |
| office_07 | 44.4% | 9 |
| living_room_04 | 44.4% | 9 |
| living_room_20 | 43.3% | 15 |
| kitchen_18 | 42.3% | 13 |
| study_09 | 41.7% | 12 |
| meeting_room_03 | 41.7% | 12 |
| living_room_02 | 41.7% | 12 |
| study_17 | 41.2% | 17 |
| kitchen_09 | 39.3% | 14 |
| kitchen_01 | 38.5% | 13 |
| study_22 | 37.5% | 12 |
| living_room_21 | 37.5% | 12 |
| living_room_15 | 37.5% | 12 |
| kitchen_05 | 36.7% | 15 |
| bedroom_12 | 35.7% | 14 |
| living_room_19 | 35.0% | 10 |
| bedroom_03 | 35.0% | 10 |
| kitchen_07 | 34.6% | 13 |
| study_05 | 33.3% | 12 |
| study_06 | 32.4% | 17 |
| kitchen_15 | 32.1% | 14 |
| gym_02 | 31.8% | 11 |
| living_room_14 | 30.8% | 13 |
| meeting_room_05 | 30.0% | 10 |
| office_06 | 29.2% | 12 |
| living_room_24 | 29.2% | 12 |
| kitchen_22 | 28.6% | 14 |
| study_18 | 25.0% | 14 |
| living_room_10 | 25.0% | 12 |
| gym_01 | 23.1% | 13 |
| study_19 | 20.0% | 10 |
| living_room_23 | 19.2% | 13 |
| meeting_room_04 | 14.3% | 14 |

## 文件说明

| 文件 | 内容 |
|---|---|
| `m3_eval_summary.json` | **最终汇总**：总准确率、by_type、by_video（最常用来出表/画图） |
| `run_report.json` | 完整运行报告：annotations/pipeline/subset/per_video/aggregate/judge/status=completed |
| `m3_eval_results.jsonl` | 逐题判分明细（每题：predicted vs expected、score、judge 理由） |
| `all_m3_qa_results.jsonl` | 全部 1276 题的原始 QA 记录（问题/答案/检索过程） |

每个视频的完整产物（episodic_memory / voice_asr / frames / faces / pipeline.log 等）在 `/data/m3_runs_yuanjian/full100-ds/<video>/`。
