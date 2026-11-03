# oGMemory 多模态文档存档

以下11份文件从 `oGMemory/docs/` 原样迁出，来源提交与 SHA-256 见
上级目录的 `migration_20261103.json`。复制校验完成后才删除代码仓中的原文件。
存档保留历史表述，不代表最新接口或当前实验结论；其中命令、路径仍相对原
oGMemory 仓库，部分旧采样/恢复说明已经过时，**实际操作请使用代码仓的统一指南**。

## 设计与实现细节

- [多模态架构](MultiModelMemory-architecture.md)
- [有界上下文](m3_bounded_context.md)
- [来源事实索引](m3_source_fact_index.md)
- [单视频人物关联](m3_person_grounding.md)
- [视频级语义整合](m3_semantic_consolidation.md)
- [部分记忆与失败处理](m3_partial_memory.md)

## 原运行说明

- [随机子集评测](m3_random_subset.md)
- [安全续跑](m3_resume.md)

## 历史实验记录

- [本地 GPU 验证](m3_local_gpu_validation.md)
- [MiniMax 对比（2026-10-23）](m3_minimax_comparison_20261023.md)
- [Omni7B 单样本（2026-10-22）](m3_sample1_omni7b_20261022.md)
