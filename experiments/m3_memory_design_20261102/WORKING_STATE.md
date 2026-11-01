# 完成检查点

技术实验、共同评分和报告均已完成；不再自动启动模型任务。
用户授权截止：2026-11-02 10:03 Asia/Shanghai；全部模型任务已在06:50结束。

## 最终共同评分

11条完整65题轨迹，715条评分记录，0评委错误。旧分数保留，不拼接不同批次。
- S：0.5846 / 0.6077，描述性均值0.5962。
- E：0.5846，65/65有效，仅一次；重复因额度16%低于门槛而未执行。
- 同期B：0.5308；旧B两条固定答案本批重评0.5923 / 0.6000。
- M3参考适配：旧轨迹与同期重复均0.4615，不能称为官方权重复现。
- F/FI/FIP：0.5000 / 0.4923 / 0.5462；FIP唯一协议失败按0保留。

E较同期B：总QA输入少38.2%，未缓存输入少63.6%；不能解释为费用同比下降。
S均值与旧B均值相同，E没有重复，不能宣称稳定全面提分。

## 产物与复查

- REPORT.md、comparison.json、joint_questions.csv/.jsonl：最终统一统计。
- docs/m3_memory_design_findings.md：结论、边界和建议。
- latex/report.pdf：16页，已编译并视检新增结果、图、分类表和限制说明。
- final_artifact_verification.json：715任务、55预测文件、13源码归档、M3输入依赖hash。
- input_audit_final.json：两个数据库内容与initial一致。
- source_audit_first_four.json、source_audit_S_repeat.json、source_audit_B_E.json：
  455题251023个可见原文位置验证通过。
- strict_delivery_audit.json：E严格3500上限、0重建、305/305消息前缀保持；
  57 required请求只有41原生Answer，16普通文本兼容，不能说原生强制保证。
- source_retrieval_replay_full.json：371/371重放一致，355选择改变，19次原候选池外节点。

## 必须保留的限制

共同评委sitecustomize启动时缺源码路径，714个需模型任务的usage/逐响应模型标识未落盘。
评分完整，QA用量完整；不能给出整轮完整总tokens/费用，不能估算补账或重评成功项补账。
详见judge_instrumentation_incident.json。后续启动路径和失败即禁止调用的预检已修复。

增量节点/片段cap是本次准入，不是全部保留历史；全局token预算覆盖全部历史。
身份条件只是文本名字/角色声明，不是已核实的人脸—声音—姓名链接。
固定记忆QA-only、current_clip、5个开发视频、多标签类别重叠；非严格因果/独立留出评测。

## 版本

oGMemory当前8626d24a，分支m3_dev_1031；最后一次代码变化547e17b7，之后只补文档。
E实际推理冻结a6abc119；S冻结a1e3160b；F/FI/FIP冻结a9b0b2dc。
完整源码与配置归档各自不变。外层主仓main最终提交以git log和交付消息为准，不push。
最终检查后应完成goal并报告；没有新的自动实验队列或待补评分。
