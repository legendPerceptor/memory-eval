# 正式五视频协议：MiniMax-M3.1-Flash / adaptive-medium

固定日期：2026-10-31。该计划在本配置第一份 QA 答案产生之前建立。

## 实验身份

- 模型：用户指定的 `MiniMax-M3.1-Flash`，端点 `https://api.minimax.cn/v1`。
- 从交互式 bash source `~/.bashrc`，密钥读取 `MINIMAX_API_KEY`，不出现在公开产物。
- 代码：oGMemory dev 固定提交 + 本目录 `source.patch`；哈希见 `provenance.json`。
- 数据库 `ogmem_m3_flash_medium_20261031`，账号 `m3-flash-medium-20261031`，
  向量表 `m3_episodic_vectors`，新库启用 vector 扩展，不导入任何前序情景记忆。
- 入口：`../../scripts/run_m3_flash_medium.sh`，仍调用原 `run_m3_random10.sh`。
- 完整时长，固定 sample-size=5、seed=42，共 65 题：
  study_05(12)、gym_03(11)、bedroom_04(11)、study_18(14)、kitchen_20(17)。

## 正式参数

| 项目 | 设置 |
|---|---|
| 记忆、QA、judge | 全部 MiniMax-M3.1-Flash |
| 推理 | thinking=adaptive，reasoning_effort=medium |
| 输出预算 | 记忆 8192、QA 8192、judge 2048 max_completion_tokens |
| temperature | reasoning 模式不发送；不能把 YAML 中的数值写成实际请求温度 |
| 超时 | 单次 HTTP 300 秒；QA 控制循环 180 秒，最多 6 轮 |
| 记忆生成并发 | 4 workers，图/向量连接池大小均为 5 |
| 视觉 | 30 秒切片，1 fps 前缀采样，至多 16 帧，512 缩放，JPEG 80 |
| 音频 | 16 kHz 单声道，原始音频 + ASR，不关闭音频 |
| 本地模型 | BGE-M3 1024 维、InsightFace buffalo_l、Whisper large-v3-turbo、pyannote、ECAPA |
| GPU | 既有 CV/embedding 服务；ASR、分离和声纹使用物理 GPU 2 |
| 检索 | top-k=8，candidate-k=20，transcript-top-k=20，RRF，context-k=8，neighbors=1 |
| 时间 | current_clip，非严格因果；逐题记录实际未来宽限 |
| 上下文预算 | 不启用 qa_context，保留原 wrapper 检索策略 |
| 刷新 | force 主要抽取阶段，禁用 smart-resume，不复用前序 LLM 记忆 |

## 为什么不用 max

用户给出的 max/500 文本示例成功，但真实片段中 max/8192 把输出预算全部用于
reasoning，没有正文。同片段改为 medium/8192 后得到完整 JSON，reasoning=431、
总输出=1451。正式评测据此选择可运行设置，不把失败样本替换成其他视频。
前序 max 实验及 Omni 试运行均已中断，不能作为完整模型质量对照。

## 工程修改与验证

1. M3 配置、模型工厂与 MiniMax provider 透传已有 reasoning 请求机制。
2. 可选独立片段并发：默认 workers=1；最多 4。并发前解析输入，拒绝重复 RawNode；
   模型调用及独立片段写入可并发，最终输出按原 manifest 顺序汇总；
   不吞掉单片段错误，仍产生失败清单并阻止依赖阶段。
3. 本轮把 workers 设为 4，只是吞吐优化，不宣称提高语义质量。
   需要实际 SQL 审计确认最终覆盖和无悬空/跨账号关系；单元测试不是实测替代品。
4. 实际执行 M3 + LLM provider + pipeline runner 测试：976 passed in 10.65s。
   外层统计脚本 9 项测试通过；完整仓库测试未在本轮全量运行。

## 指标与报告规则

- 固定 65 题，不能仅用成功回答作为分母。
- 报告每视频/题型的记录覆盖、有效答案、记忆/向量覆盖、失败阶段、
  `{0,0.5,1}` 题目加权均分和视频宏平均。
- 缺失答案/执行失败在端到端效用分计零并单列；judge 故障不解释为模型答错。
- 同模型评分只是开发诊断，不是独立人工准确率；study_05 是历史开发样本，
  本轮不是独立留出测试，不推导全数据集泛化增益。
- 原始调用、重试、恢复分开保存；不按最高分拼接答案；变更配置另立实验身份。
- 完整调用元数据记录模态、模型、tokens、thinking、effort、finish_reason、耗时，
  不包含提示/媒体/API 密钥。并发请求的耗时和不能当作整轮墙钟时间。
- 视觉前缀采样遗漏、current_clip 时间宽限、默认 QA 未直接消费所有图结构，
  必须在报告中保留。

## 可复现性边界

seed=42 仅固定视频抽样；远端生成未固定 seed，也没有发送 temperature，
不能保证再次运行的答案逐字相同。保存实际响应、模型名、时间、用量、源码、
本地依赖版本和视频 SHA-256，不宣称锁定了供应商后端权重版本。
