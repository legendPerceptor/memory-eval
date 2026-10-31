# MiniMax-M3.1-Flash 五视频评测

## 变更原因与实验身份

2026-10-31，用户要求停止原 Omni-7B 实验，改用
`https://api.minimax.cn/v1` 的 `MiniMax-M3.1-Flash`。
已在交互 bash 中显式 source `~/.bashrc`，使用 `MINIMAX_API_KEY`；
密钥不写入报告、命令参数或公共日志，配置使用环境变量引用。
原 `m3_baseline_20261031` 保留为 **用户要求切换模型而中断的试运行**，
没有完整 QA 成绩，不能用来计算新模型提升幅度。

## 协议

- 相同 seed=42、相同五视频和 65 题，完整时长。
- `study_05` 12、`gym_03` 11、`bedroom_04` 11、`study_18` 14、`kitchen_20` 17。
- 新数据库 `ogmem_m3_flash_20261031`、账号 `m3-flash-20261031`、
  向量表 `m3_flash_20261031`，不复用 Omni 生成的记忆。
- 记忆、QA、judge 均使用用户指定的 MiniMax-M3.1-Flash。
- 所有请求 `thinking={"type":"adaptive"}`、`reasoning_effort="max"`；
  开启 `is_reasoning_model` 后发送 `max_completion_tokens`，不发送 temperature。
  YAML/CLI 保留的 temperature 不应误报为实际推理模式的采样设置。
- 记忆与 QA 输出预算均为 8192 tokens；judge 为 2048；
  HTTP 请求超时 300 秒；QA 控制循环为 6 轮、180 秒。
- 输出预算区别于原 Omni 的 1024/4096/4096；因此更不能把两组称为
  仅替换模型且预算一致的严格对照。用户的 500 tokens 仅用于文本连通性测试。
- 其余媒体、检索与时间策略沿用原 wrapper：30 秒、1 fps 前缀至多 16 帧、
  原音频 + ASR，top-k=8、candidate-k=20、transcript-top-k=20、
  RRF、context-k=8、neighbors=1、current_clip，不开启 qa_context。
- 本地 GPU 继续提供人脸与 BGE-M3 服务；faster-whisper、pyannote 和
  ECAPA 在 `CUDA_VISIBLE_DEVICES=2` 执行。声纹显式 `cuda:0` 指该可见卡，
  避免 SpeechBrain 对字符串 `cuda` 的解析告警；没有切换到物理 GPU 0。

## 兼容修改

原 M3 配置/工厂没有透传推理参数，MiniMax 构造函数也没有接收入口。
本次仅增加可选 `reasoning_effort`、`is_reasoning_model`、`extra_body`，
通过已有 OpenAI-compatible 请求参数机制传递；旧配置未设置时行为不变。
源码补丁与新增回归测试保存在本目录，以固定提交加补丁复现实验。

实际执行 `tests/unit/m3 tests/unit/providers/llm`：
最终 **963 passed in 10.60s**。首次测试的 3 个兼容失败及修正过程日志保留在
`private/provider-tests.log`，未把该次失败写成通过。

## 预检与统计

1. 用户提供的文本请求实际返回 HTTP 200，响应模型名匹配，回答正确。
2. 使用已提取的一个真实片段检查完整 16 帧 + WAV + ASR 输入，不写入评测库；
   该预检只是接口兼容检查，不算入五视频记忆成绩。
3. 实际测试原生工具调用；通过后才启动完整运行。
4. 完整调用审计只保存模型、用量、模态数量、finish_reason、耗时、
   推理参数等元数据，不保存密钥/提示/媒体。
5. 以固定 65 题为分母；缺失/失败与 judge 故障区分；不挑选最佳重试答案。
6. 同模型评分是诊断性 `{0,0.5,1}` 均分，不是独立人工准确率；
   `current_clip` 不是严格因果；采帧不是每片段全时段覆盖。

预检、本轮与恢复若发生，各自分别记账。SDK 内部 HTTP 重试不等同于审计中的
逻辑 completion 调用数；token 汇总不是账单金额。
