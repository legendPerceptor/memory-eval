# M3 有界检索上下文

## 目的和兼容性

旧 M3 QA 限制检索条数和模型轮数，但不限制整个请求的输入大小。
不同查询返回重叠片段时，完整 description / events / transcript 继续累积。
默认 top_k 也不是服务端上限：模型可以请求更多主结果，再叠加邻居。

本功能通过 `m3.qa_context.enabled` 显式启用，默认关闭以保留旧实验可复现性。
ReAct QA driver（`main_react_qa`）、pipeline 的 QA stage、eval 的在线 QA 路径
读取同一 YAML 配置；
judge-only 路径不修改已生成的答案。直接构造 adapter 的代码可传入
`context_budget=ContextBudget(...)`。
注意：直接 `python -m m3.qa` 仍是旧单次 driver，不是本功能的 ReAct 入口；
原 `scripts/run_m3_random10.sh` 使用的 pipeline QA stage 则已接入。

本文默认描述每次重建完整投影的模式。后来新增的实验性
`incremental_delivery`另见`docs/m3_source_fact_index.md`：
其中入选节点/每节点片段上限作用于本次新增准入，并非整个保留历史的全局上限；
全局输入token预算仍覆盖全部历史。不要将两个范围混淆。

## 在当前版本中生成配置副本

不必修改原模型配置，也不要覆盖历史实验的私有配置：

```bash
python -m m3.configure_qa_context \
  --config /path/to/your/ogmemory.yaml \
  --output /path/to/private/ogmemory.bounded.yaml \
  --profile balanced \
  --tokenizer /path/to/pinned/tokenizer.json
```

- `balanced`：24000输入目标（若QA输出8192、窗口32768、安全预留2048，
  有效输入为22528），原文passages格式，45秒回答预留。
- `compact`：12000输入目标、passages格式；小预算可能损失质量，不是默认升级。
- `grouped`：12000输入目标、分组原文格式；应依据独立评测选择，不保证质量提升。
- 可传 `--expected-tokenizer-sha256` 校验本地tokenizer；不联网下载。
- 工具只改变新副本的`m3.qa_context`，不改变模型、输出预算或数据库/凭据设置，
  不执行模型或存储调用。副本权限为0600，已有文件不覆盖，错误不打印配置内容。
- 副本含原配置的私有连接信息，**不要提交Git**。
- 与本轮实验匹配的ReAct循环为6轮、180秒；这是运行参数，不由此配置工具改写。
  32768是所选运行预算，不是对任何模型原生上下文容量的断言。
- **使用新的QA结果文件/实验目录**。不要在更换模型、上下文策略或时间策略后，
  对旧输出直接使用`--resume`，否则可能沿用旧答案而不是执行新方案。
- 此工具用于当前代码版本；不要给旧冻结源码添加它不认识的新配置字段。

下面是可直接并入已有模型配置的等价字段示例；模型角色仍沿用原设置。

```yaml
m3:
  qa_context:
    enabled: true
    context_window: 32768
    max_input_tokens: 24000
    safety_tokens: 2048
    max_top_k: 16
    max_candidates: 32
    max_result_nodes: 24
    max_neighbor_clips: 64
    max_search_calls: 8
    max_selected_nodes: 32
    max_passages_per_node: 32
    # 可选：为回答留时间，0 保留旧控制器行为；必须小于 QA timeout_seconds。
    # answer_reserve_seconds: 45
    # 可选：按字段分组，减少已入选原文的JSON包装；默认passages保持历史格式。
    # evidence_format: grouped
    # 安装 m3-budget extra 后可指定本地、已校验的 tokenizer JSON，不联网下载：
    # tokenizer_path: /path/to/pinned/tokenizer.json
    # tokenizer_multiplier: 1.15
  qa_llm:
    # 其他 provider / model / 凭据设置保持原样
    max_tokens: 2048
```

输出预留读取 `m3.qa_llm.max_tokens`，不在 qa_context 中重复配置。
有效输入上限为：

```text
min(max_input_tokens, context_window - max_output_tokens - safety_tokens)
```

## 三层控制

### 1. 服务端检索限制

- 模型请求的 top_k 被限制到 `max_top_k`。
- 向量分支及 transcript 分支各自不超过 `max_candidates`；
  模型不能再通过增大 top_k 无限扩大候选数。
- `max_result_nodes` 包含主结果、邻近片段和时间边界片段，不是三个独立预算。
  保留边界证据槽位，再分配主结果和邻居。
- `max_neighbor_clips` 独立限制每次 Search 实际请求的邻居 clip 总数（默认 64），
  不与最终 `context_k` 混用。按距离、主命中排名顺序分批读取，每批最多 8 个 clip，
  且不超过尚缺的邻居证据数。空 clip 不消耗结果槽位；获得足够有效证据、
  用完读取预算或遍历完 `include_neighbors` 范围后停止。
  `max_result_nodes` 不再缩短邻居半径；读取预算耗尽时仍可能返回少于 `context_k` 条。
- 每题至多 `max_search_calls` 次可执行的 Search 请求；缓存/无效请求也消耗
  该调用额度。超过额度只返回拒绝信息，不继续检索，并切换为 Answer-only。
- 查询大小受 `max_query_bytes` 限制，超长查询拒绝，不偷偷改写为另一条查询。

账号、视频、时间范围仍由原检索服务校验，不从模型传入的引用猜测权限。

### 2. 按内容版本去重和原文片段选择

`m3/context_budget.py` 每轮从完整工具历史重新建立模型视图：

- 证据身份包含节点、视频、片段和完整内容投影的 hash；排名分数等不参与 hash。
  相同 ID 的内容更新、不同时间边界投影不会被误合并。
- 不同查询返回相同证据时，只在首次出现的工具结果中放正文，其他出现位置放引用。
- 优先将短片段的完整 description 与连续转写一起选入，保留局部叙述的连续性；
  超大叙述退回连续原文窗口，而非全局打散为独立句子。补充事件、人物和实体
  排在主要叙述之后，减少重复表达挤占关键转写的容量。
- description 保存原文字符区间；转写、事件、人物和实体保留原始列表项和字段下标。
  不调用另一个 LLM 总结，不补写事实。
- 依据问题、最近检索词和原检索排名选择预算内的原文组合。
  单节点扫描片段数、入选节点数、每节点入选片段数都有上限。
- 每次构建上下文最多进行 64 次完整输入试装计数，剩余不足 128 个计数单位时
  提前停止；不为填满最后一点容量反复做整段 tokenizer 编码。
- 所有引用只指向**当前请求内实际保留**的证据；下一轮重新选择并重建引用，
  不依赖已被移出上下文的旧正文。
- 工具结果明确告知内容不完整，遗漏不是事实不存在，检索到的事件数不代表完整总数。

完整工具结果、cache 和 `tools_used` 不被模型视图覆盖；QA 的持久化
`search_stats.context_calls` 记录入选来源、内容 hash 和原文范围，而不重复输出
整段正文。原始记忆仍在数据库中，可用来源信息复核。

当前还没有独立的 ReadEvidence 工具，也没有跨视频分层索引；这是第一阶段，
不是完整的长时视频记忆架构。后续可增加受权限和时间范围约束的按需展开接口。

### 3. 完整请求检查

计数覆盖 system/user 提示、历史 assistant/tool 消息、JSON 转义和工具 schema，
再增加消息 framing 预留；不是只统计 evidence 正文。
基础提示和控制历史已超过预算时，直接返回明确的预算错误，**不发送模型请求**。

默认计数器使用“序列化 UTF-8 字节数＋framing”作为保守估计，而非字符数除以 4。
可选的本地 tokenizer 模式需要 `m3-budget` extra，读取已固定版本的 tokenizer JSON，
禁止自动下载，并关闭该 tokenizer 保存的 truncation/padding，避免计数时先截断输入。
对完整序列化请求计数后乘以保守系数（默认 1.15），再增加 framing 预留。
入选诊断记录 tokenizer 文件 SHA-256。

**两种方式都不是任意远端模型/模板的数学保证**：序列化请求计数并不等同于
服务内部 chat template 的精确计数，Qwen tokenizer 对 MiniMax 只能作为代理。
实验必须同时检查服务返回的实际 `prompt_tokens`，未知模型或模板不能仅凭
本地估计宣称绝不会越界。没有合适 tokenizer 时可回退到字节模式，但更保守的
预算可能挤掉关键证据；这同样必须通过质量对照检查。

## 诊断与验证

### 可选的分组原文格式（2026-11-01）

`evidence_format: grouped` 只改变模型视图中的表示法，不生成新的摘要：

- 保留源节点、video/clip和时间元数据；
- description表示为 `[start_char, end_char, original_text]` 的列表；
- speech_transcript/events/characters/entities表示为
  `[original_list_index, original_value]` 的列表；
- 原始value中的说话人、时间、置信度、嵌套空间关系等字段全部保留；
- `field`/`index`/`value`不再在每条原文外重复包装；
  内容hash不再占用模型输入，但完整保留在`selected_sources`审计中；
- 字段内的入选顺序不变；角色/邻居关系和有效证据引用仍保留；
- 系统提示和工具提示解释数组含义，特别注明下标不是动作次数。
- 下标是**单个片段、单个字段内部**的位置，不是跨片段的人物或物体ID；
  不可依据两个片段都有索引0就合并身份。它与M3-Agent的实体锚点ID不同。

这不是“整个检索结果无损压缩”：**已入选原文的表示是无改写的**，
但整个流程仍有预算选择与遗漏。分组节省容量后，贪心选择可能容纳更多
原文，因此需要独立QA质量对照，不能仅凭字符减少声称效果提升。
默认passages保持历史模型输入不变。

### 可选的回答收尾（2026-11-01）

`answer_reserve_seconds > 0` 启用 M3 专用收尾，不改变通用 ReactLoop：

- 在每轮模型调用前检查剩余时间；不足“回答预留 + 上次模型耗时”时，
  不再开放 Search，只开放 Answer。预留不是额外赠送的时间。
- 最后一轮同样关闭 Search，不再只是发送“不要调用工具”的提示但仍暴露 Search。
  模型若忽略工具 schema，服务端也拒绝新的 Search，不延长轮数。
- 与 M3-Agent 最后一轮要求回答的思路相近，但**不要求合理猜测**：
  回答必须有证据，信息不足时明确说明。
- 这是软预算的提前收尾，不是 HTTP 的硬取消。一次极慢请求仍可能越过
  180 秒；没有补跑/增加轮次来伪装完成。上游通用 operation deadline 不变。
- 默认 `0` 保留历史实验；启用时，值必须小于 adapter 的 timeout_seconds。
- 上下文压缩保留工具的 `guidance` / `duplicate_query`，避免丢失无新增证据提示。
- `search_stats.finalization_reason` 记录 `answer_time_reserve` / `last_iteration`；
  每轮 context_calls 记录打包前剩余时间、answer_only、model_seconds。
  这些耗时不含服务端无法观测的内部细分，不视作费用统计。

8192 输出预算搭配本页示例32768窗口和2048安全预留时，实际输入上限是
**22528**，不是24000。不要通过降低输出预算掩盖推理模型没有正文的问题。

每轮 `search_stats.context_calls` 包含：

- 计数方式、估计输入、有效上限及未压缩历史大小；
- 返回节点出现次数、唯一内容版本数、重复次数；
- 入选节点和片段数、被跳过的超大节点数；
- 每份入选证据的节点/视频/片段、内容 hash、字段/下标或字符范围。
- `visible_passages` / `new_visible_passages` 统计实际交付视图中的原文范围，
  按内容版本及范围区分；不把检索到但未入选的节点算成已读。
  `search_requests_seen` 将本次输入与之前的检索次数对应起来。
  当前此项仅用于审计，不更改 no-gain 控制策略；问题之间清空状态。

验证时应同时报告：

1. 每个实际模型请求的输入及输出预留是否在服务上下文容量内；
2. 服务端 top_k、候选数、总返回数和 Search 执行次数是否受控；
3. 引用可解析、时间边界与账号不串漏，原始记忆及工具审计不被改写；
4. 答案覆盖、失败类型、固定评委的逐题评分，而不是只报告 token 缩减；
5. 小样本上的质量变化不能作为大规模质量保证，尤其关注计数题、空间关系和人物归属。

## 单视频验证（2026-10-23）

使用 `study_05` 的同一份 69 条 MiniMax 记忆、同一数据库、同一组 12 题。
只运行 QA，不重新采帧、转写或抽取记忆。

- MiniMax 旧模式与有界模式：相同模型、6 轮、180 秒、2048 输出 tokens、temperature=0。
- 本地 Omni-7B 有界模式：仍读取上述 MiniMax 记忆，不能称为本地端到端记忆重建。
- 三组均由相同 MiniMax 评委、相同 1024 输出预算评分，失败题计 0。
- 基线、每次失败和模型用量日志分别保留；不通过挑选成功重试覆盖首轮。

结果和可重跑脚本位于：

```text
deploy-llm-on-gpu/results/m3-bounded-context-sample1-20261023/
```

数据库在实验前后计算节点、边及向量表的完整行哈希，以确认没有修改记忆。
自动评分只是诊断，不是独立核实的准确率。

### 最终结果（v4，保留全部 12 题）

| 模式 | 有效答案 | 实际输入峰值 | 累计输入 tokens | 同一 MiniMax 评委均分 |
| --- | ---: | ---: | ---: | ---: |
| MiniMax 旧模式，本轮重跑 | 11/12 | 138546 | 3089687 | 0.4583 |
| MiniMax 最终有界模式 | 11/12 | 18095 | 919526 | 0.5417 |
| 本地 Omni-7B 最终有界模式 | 12/12 | 11978 | 111434 | 0.1667 |

MiniMax 峰值输入下降 **86.9%**，累计输入下降 **70.2%**。
这不等于账单下降比例，也不是吞吐提升保证：最终 MiniMax 调用了 66 次，
旧模式为 56 次。模型如何检索、何时回答也随可见证据改变。
没有单独跑同一份 MiniMax 记忆上的本地旧模式，所以本地一行只证明可运行性，
不能据此计算本地模型的新旧策略质量增益。

最终配置在上面示例基础上指定本地 Qwen2.5-Omni-7B tokenizer：
revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`，倍率 1.15，
最大 32 个入选节点、每轮最多 64 次试装。tokenizer 来自已有模型缓存，
没有下载新模型；实验 Python 环境的 tokenizers 为 0.23.2。
**不是只打开 enabled、保持字节计数，就能复现最终结果。**

最终 66 个 MiniMax、24 个 Omni-7B 请求均核对了实际 usage：

- 实际输入均不超过本地估计，也不超过 24000；
- 实际输入加请求的 2048 输出预留均小于 32768；
- 所有 Search 的实际条数、返回节点总数、执行次数均符合服务端限制；
- MiniMax 打包耗时中位数/最大值为 1.765/2.522 秒；
  Omni-7B 为 0.460/1.287 秒，没有将 tokenizer 的 CPU 成本隐藏在模型耗时中。

MiniMax Q12 仍失败：输入 **17775**，输出 **2047**，其中 reasoning **2047**，
`finish_reason=length`。不是输入 context-window 超限，也未通过增加输出预算
或挑选重试来补成 12/12。原样保留失败，计分为 0。

### 不能仅看均分

参考分未下降，但不能把它当成独立确认的质量提升：

- 树干步骤的完整叙述和连续转写已恢复，clip 5 的计数语句实际交付给模型；
- clip 52 的黏土描述和两条转写也实际进入 Q09 的多轮上下文，
  最终回答仍没有准确保留材料信息，说明还有证据整合问题，不只是召回/压缩问题；
- 书包回答仍说桌子右侧，而金标为左侧，评委却给了半分；
- 树的数量仍把“再做一棵”的计划当作已完成；最后一题输出推理耗尽；
- 本地 7B 虽然能跑完，质量仍低，不能宣布整体质量验收通过。

结论是 **上下文容量控制验证通过，整体答案质量尚未验收通过**。
该视频是开发样本，期间依据失败现象调整过实现，不是独立留出测试集。
下一步需固定版本，验证更多视频和计数/材料/时序问题，不能继续只对本样本调分。

### 保留的试验过程

| 版本 | 变化 | MiniMax 参考均分/状态 |
| --- | --- | --- |
| v1，结果根目录 | 字节预算，独立片段排序，召回上限较窄 | 0.2917，质量退化 |
| v2 | 连贯描述＋转写，恢复召回宽度，仍用字节预算 | 0.3333，仍低于基线 |
| v3 | 引入本地 tokenizer、扩大有界工作集 | 因反复整段计数的 CPU 开销中断，未作为完整质量对照 |
| v4 | 保留 tokenizer 和连贯证据，限制试装次数 | 0.5417，最终完整对照 |

各版本使用同一份固定记忆；v2/v4 复用根目录的本轮基线答案和评分，
不是额外独立的基线重复实验。没有将不同版本的最佳单题拼成最终结果。
v1/v2 完整结果、v3 部分输出、各版源码快照及 v4 全部结果均保留。

最终机器报告：`v4/comparison-report.json`；逐题答案和评分理由：
`v4/answer-comparison.json`。脚本输出路径相对脚本目录，复跑应先复制到新结果目录，
不要覆盖原实验记录。

### 验证与部署边界

- 完整单元测试：**7893 passed、38 skipped、15 warnings，无失败**。
- M3 人脸检测集成测试：**13 passed**；相关未定义名称静态检查通过。
- `uv lock --check --offline` 通过；新增可选 `m3-budget` 依赖，没有同步或替换现有 GPU 环境。
- 数据库 5679 节点、7420 关系、69 向量的完整行哈希前后一致。
- 全局私有配置未改，locomo-test 未改，服务未重启。
- 默认仍关闭；实验私有配置已启用。未自动切换线上服务或宣称大规模压测通过。
