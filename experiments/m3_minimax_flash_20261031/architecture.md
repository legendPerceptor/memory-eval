# 当前方案的代码级梳理（MiniMax 主实验）

以下陈述来自本地固定版本的实现，不将历史文档里的成绩当作本次成绩。
所有源码路径相对 `oGMemory/`。

## 1. 输入与证据层

`config/m3/pipelines/video_episodic_full.yaml` 定义顺序执行的 18 个阶段。
视频以 30 秒为单位切片，建立 VIDEO RawNode、带来源关系的 IMAGE / AUDIO
子节点。帧提取每秒采样一次、最多 16 张，而不是全片段等距 16 张。
完整音频先做 Whisper 转写，保留词段时间；pyannote 分离说话人，
SpeechBrain 提取 192 维 ECAPA 声纹。HTTP CV 服务做人脸检测与特征提取。

主要实现：`m3/preprocess.py`、`m3/load_clips.py`、`m3/extract_frames.py`、
`m3/extract_audio.py`、`m3/pipeline/stages.py` 与 `extraction/operators/`。
数据集自带的 `memory_graphs` / `intermediate_outputs` 不是本次新记忆的来源；
本次从视频重建，`force=true` 刷新主要抽取环节。

## 2. 情景记忆与结构化存储

`m3/episodic_memory.py` 将片段画面、可用原始音频和时间标记的 ASR 文本交给
`EpisodicMemoryGenerator`。输出包含 description、characters、events、entities，
存为视频对应的 ExtractedNode，并记录片段来源时间。空音频不强行附加，
被模型截断的响应不能靠补 JSON 括号冒充完整成功。

随后创建具有来源支持关系的片段 MemoryNode。人脸聚类、声音聚类建立身份关联；
这些阶段是否为零产出、是否发生 fallback，应以本次阶段诊断为准。
`m3/semantic_memory_consolidator.py` 按片段顺序拼接 description（上限 50,000
UTF-8 字节），对人物、事件、实体进行确定性去重合并，产生视频级 MemoryNode，
同步 SUPPORTS / EVOLVES 来源关系。
这是规则式聚合，不是新增 LLM 驱动的抽象推理或事实冲突消解。

## 3. 索引与检索

`m3/index_episodic.py` 将结构化情景内容与原始时间标记转写展开为文本，
调用 BGE-M3 生成 1024 维嵌入，写入 SQL 图及独立向量索引。
`m3/control_adapter.py` 默认限制 extractor 类型为
`episodic_memory_generator`，因此默认 QA 主检索的是情景 ExtractedNode。
语义汇总 MemoryNode 和身份关系的生成，不等于本轮主检索直接消费这些节点/边。

Search 的实际检索路径包括：

1. 对原查询做向量召回；人名识别后可追加去人名查询，以减轻转写姓名变体影响。
2. 对合并候选做基于 transcript 的 BM25 重排。
3. 独立的 transcript 词法召回，与向量结果以 RRF 融合。
4. 保留主证据，并按片段来源查找相邻片段；必要时附加时间边界转写。
5. 保留账号、视频和问题时间约束；返回 description、人物、事件、实体、
   ASR 与 provenance，供回答引用。

默认 top-k 是模型可请求的默认值，不是绝对服务端上限。
本次未启用 `m3.qa_context`，因此不能宣称已实施整个对话的 token 上限。
已有可选实现位于 `m3/context_budget.py`：服务端条数约束、内容版本去重、
原文片段选择和完整请求计数。后续宜作为独立 QA 消融，而非默默加到基线上。

## 4. 时间边界与回答

`m3/temporal.py` 优先使用 annotation timestamp，并用实际 source interval
解析本地片段，而非假定数据集 before_clip 与本地 30 秒切片编号一致。
`current_clip` 将可见证据截止点放宽到该片段末尾：可能看到问题之后的局部信息，
所以结果不能表述为严格在线/因果记忆能力。

`m3/qa.py` 的 ReAct 入口经 `OGMemoryControlAdapter` 调用 Search / Answer，
最多 6 轮。Search 缓存可去除完全相同查询的重复正文，但不同查询的重叠内容
仍可能累积；若不启用可选上下文预算，长视频问答有输入超限风险。
180 秒是控制循环预算，单个模型请求 timeout=300 秒且 SDK 可重试，
所以它不是端到端硬超时保证。无答案、协议脚手架、模型截断等均应单列。

## 5. 评委与统计

`m3/eval.py` 将预测与标准答案交给 judge，打 0 / 0.5 / 1。
空答案直接记零，不发 judge 请求；judge API 异常通过 judge_error 单列。
本次记忆、回答、评分都使用 MiniMax-M3.1-Flash，同模型评分只用于开发诊断。
标准答案仅用于问题清单/评分；没有向记忆生成或回答提示注入 gold answer。
后续结果审计将以固定 65 题为分母，避免只统计成功产出的题目造成虚高。

## 6. 建议的后续对照（尚未实施，不能写作已验证增益）

- 先解决可运行性：模型名实时预检、数据库扩展预检、记忆输出截断统计。
- 视觉覆盖：固定相同帧数，对比前缀采样与全片段等距采样；再考虑动作自适应采样。
- 上下文：冻结本次记忆，仅切换已有 `qa_context`，报告质量、输入量、超限率。
- 记忆质量：对位置、材料、动作完成状态保留明确时间与对象证据，区分意图/实际行为。
- 计数题：建立事件去重与时间轴，而不是把检索到的事件条数当作全视频总数。
- 图价值：分别启用/关闭身份关联与视频级聚合的检索，验证它们确实影响证据和答案。
- 评分可信度：固定独立评委并抽查错误/半分题；跨视频扩大样本后再主张普遍改进。
