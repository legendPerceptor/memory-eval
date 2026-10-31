# 未完成：max 推理强度故障诊断

本次配置为 MiniMax-M3.1-Flash、adaptive/max、记忆 8192 输出预算、串行。
18:04 启动五视频管线；首个视频 `study_05` 的首片段实际调用返回
`finish_reason=length`，completion=8192、reasoning=8192，没有记忆正文。
在后续调用进行中主动停止实验，防止重复消耗。不存在整组 QA 成绩。

## 真实预检与诊断

| 调用 | 推理强度 | 输入 tokens | 输出 tokens | reasoning | finish | 秒 |
|---|---|---:|---:|---:|---|---:|
| bedroom_04 clip 0 预检 | max | 4740 | 7279 | 5782 | stop | 192.28 |
| study_05 clip 0 正式尝试 | max | 4805 | 8192 | 8192 | length | 109.18 |
| study_05 clip 0 相同输入诊断 | medium | 4805 | 1451 | 431 | stop | 85.66 |

相同输入诊断使用同一数据库中的 RawNode、同一帧/音频清单和同一 ASR 读取函数，
只改变 reasoning_effort，不写入记忆库。它只证明该片段在 medium 下能生成完整输出，
不证明 medium 在整个 benchmark 上优于 max。时延受远端负载等因素影响，
不能据一次请求声称稳定加速比例。

16 帧 + WAV 多模态兼容、模型返回名及原生工具调用均已确认。
首次预检脚本误用 `name` 而非 provider 返回的 `tool` 字段做断言；
修正脚本并验证已保存的真实响应，没有重新调用模型或伪造工具结果。

停止时未完成的 HTTP 请求不一定有 usage 回执，所以现有审计 token 总量只是
已收到响应的用量，不代表所有已计费请求。退出码 143 是主动 SIGTERM。

正式完整评测另建 `../m3_flash_medium_20261031/`：
adaptive/medium、4 个独立片段并发、新数据库，重新生成全部记忆。
