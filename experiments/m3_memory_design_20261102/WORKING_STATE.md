# 工作检查点（运行中；最终以REPORT.md为准）

更新时间：2026-11-02 04:28 Asia/Shanghai。**用户截止时间10:03**。
本轮goal仍active，不要重建goal，不要覆盖已冻结实验或重跑成功答案择优。

## 最新覆盖信息（优先于下方早期检查点）

- **04:29补充：HEAD为547e17b7**，只新增构建CLI的严格只读事务/RLS/视频过滤；
  E的推理代码仍精确冻结a6abc119，未受此构建工具改动影响。
  Root最新72项通过，`strict-final-expanded-tests.log`正在/即将完成最新HEAD全回归。
- oGMemory HEAD现在是 **a6abc119**（m3_dev_1031），此前新增通用构建CLI为c3f362fe。
  CLI真实dry-run340节点/7965窗口；重放已保存向量时与原固定索引字节一致，无新embedding/chat调用。
- **四组已完整结束**：FIP 0.530769/64有效，F 0.523077/65，FI 0.492308/65，
  S-v2 0.569231/65；均为初评，最终共同评委未运行。
- **M3同期重复已结束**：m3_reference_repeat_20261102，65/65有效，302调用，
  输入551699、缓存354700；没有初评，已加入methods.json。冻结源文件前后hash不变，
  65题及时间范围与原B匹配。不是重新生成M3记忆。
- 全量S查询离线重放已完成：371/371复现，355改变主片段选择，
  19次引入原候选池外节点。只说明检索路径，未用gold，无chat调用。
- 第一批四组139712个可见原文位置全部审计通过；两个数据库指纹仍与initial完全一致。
- 03:34发现并修复**名义增量计数基数受被丢弃检索审计字段影响**：
  1000代理单位设置下，旧代码可准入7095；总输入上限仍正确。
  修复后同fixture有/无审计字段模型视图一致，均只增加890。
  新代码从实际空证据模型投影计数，不会因为无新Search的回答请求余量小而重建。
- 未运行I/H/H-repeat旧快照已标记superseded_before_execution，没有调用；
  **不要再尝试运行这些旧组**。新可选组为E和E-repeat：
  `m3_strict_incremental_20261102` / `m3_strict_incremental_repeat_20261102`。
  冻结a6abc119；3500严格增量、fact/identity/index、grouped12条、required收尾，
  不加scene anchors。全局22528、6轮180秒、输出8192不变。
- E仅共同评分（skip_initial_judge），所以metrics初评分应为null，不能当成0分。
  每题前同账户只读额度检查，<=12%或查询失败即停止；只读查询最多重试3次，
  不重试任何模型成功答案。停止码76=stopped_quota_reserve，未尝试题无伪造行。
- 后续队列仍先S独立重复，再精确旧源码的同期B，然后资源允许才E/E-repeat。
  S启动30%、B40%、E30%，E重复可按上一次E消耗+12（最低30）判定；
  任何可选组仍需至少剩2小时，**不以分数决定是否重复**。
  共同评委启动门槛现在10%，E的12%逐题预留保护它。
- 最新调度进程：**session 91168**（followup，等待05:01），
  **session 32259**（common judge等待completion）。旧2649/89036等已停；
  M3重复56646也已完成。需要时ps核对，不要杀正在跑模型的进程。
- 当前源代码严格预算unit selection **1099 passed**；root最新记录70 passed，
  之后又增两项summary测试，需复跑。扩展1440 passed/108 skipped/9 deselected
  是c3f362fe；a6abc119最终扩展还需跑，预计1446但不能未跑就宣称。
- 主仓最近已提交仍e8e536f，之后大量结果/工具/文档改动尚需checkpoint与最终提交。
- renderer已支持M3重复、E/E重复、QA-only缺失初评null和按题平均重复的探索性对照。
  所有组具体对照方写在contrast字段，避免把对B的区间误称对M3。

## 已完成

- oGMemory当前提交 `39fb6392`，分支 `m3_dev_1031`；所有新开关默认关闭。
- 主仓最近提交 `e8e536f`（之后分析脚本/状态有新增更新），分支main。
- F/FI/FIP源码均冻结a9b0b2dc；S-v2/S-repeat冻结a1e3160b；
  I/H/H-repeat冻结39fb6392；同期B精确复用前次B归档/配置。
- 旧S/S-repeat（0256e574，2:1融合）在运行前废弃，无模型调用；保留清单。
- 最新相关M3/控制测试85项通过；此前扩展1422通过/108跳过/9个已知失败排除。
  39fb6392最终全回归尚应重跑（预计新增测试后总数更高，不能直接宣称旧数代表新HEAD）。
- BGE-M3来源事实索引7965窗口、340片段、37MB、约63秒；原文hash与范围审计全通过。
- FIP全65题的39599个可见原文位置审计通过，无时间越界/重复位置。

## 完整初评分（不是最终共同评分）

| 组 | 有效 | 初评分 | QA输入 | 缓存输入 |
|---|---:|---:|---:|---:|
| FIP | 64/65 | 0.530769 | 3314963 | 1776756 |
| F | 65/65 | 0.523077 | 2560063 | 511783 |
| FI | 65/65 | 0.492308 | 2548205 | 545735 |

FIP唯一失败study_18_Q03是末轮Search文本协议，33秒结束，非超时，保留0分。
两个非benchmark工具选择探针证实required/指定Answer均返回原生调用；另计1172 tokens。
探针答案解释有问题，**只能证明协议支持，不能证明推理质量**。

## 正在运行的队列

初始三组调度已完成。
`scripts/continue_memory_design.py` 正运行S-v2，随后S-repeat、同期B。
核心状态：`followup/coordinator_status.json`。
同进程随后按额度/时间执行可选I、H、H-repeat；完成的可选组才在共同评分前加入methods.json。
规则：至少剩2小时；I/H需40%额度；H-repeat同窗口可用
max(30, H实际消耗百分点+17)，否则40%。不读取得分来决定运行/重复。
S核心门槛30%，B核心40%。MiniMax当前窗口00:00–05:00，之后05:00–10:00。
01:57 S启动前额度36%；**额度不是总输入tokens的线性函数**，不要盲目预计可跑全部可选组。
无需购买或切换API key/模型。

后续完成标志：`followup/completion.json`，不是核心coordinator临时finished。
`scripts/finish_memory_design.py` 等待该标志，然后额度>=15%、至少剩1小时才启动共同评委，
自动运行renderer；最终状态在`judging/coordinator_status.json`。
共同评委是`m3_memory_design_joint_judge_20261102`，seed1102，全部方法/问题交错、
方法名不发送给评委。原B两次与M3参考适配的固定答案也共同重评。

活跃工具session检查点：后续队列2649；最终评委等待89036。
可用ps查实际进程。**不要杀正在执行模型的调度进程**；先查状态。
初始队列97242已退出，FIP单独driver76409也已结束。

## 分析脚本与待完成

1. 持续监测S/S-repeat/B/可选组状态、额度和失败；完整组用
   `scripts/summarize_context_experiment.py NAME`汇总。
2. S完整后可运行`replay_source_fact_retrieval.py --max-searches 20`，
   再用`--max-searches 0 --output source_retrieval_replay_full.json`做全量：
   读取S冻结代码，同查询比较有/无事实索引，回源校验；仅本地embedding/SQL，
   无聊天模型、不用参考答案。尚未执行。
3. 完整新组用`audit_memory_design_sources.py NAMES... --complete --output ...`检查
   所有实际可见来源，已包含I/H与老组格式；不改变答案。
4. `audit_memory_design.py final`生成数据库指纹，与initial及前次基线指纹比较。
   还应验证canonical标注哈希、归档sha、索引sha与各重复配置一致。
5. 共同评委若报错，保留原记录，只补API错误，不能重评成功项择优。
   renderer当前读取grades.jsonl；如确有恢复，要明确增加恢复来源和审计，不能覆盖原评分。
6. `render_memory_design.py`已准备：最终表/逐题CSV JSONL/REPORT.md/LaTeX结果节；
   当前只计算相对同期B的video-cluster bootstrap，**不要误称为相对M3的区间**。
   可补充FI-F、FIP-FI、S重复均值-FIP等明确标注的探索性成对差异。
7. `plot_memory_design.py`在comparison.json完成后输出向量图，尚未执行。
   `memory_design_call_ledger.py`统计全部11月2日实际记录（包含未完整可选组），探针另计。
8. 最终需人工写分析结论/案例边界，更新docs/m3_memory_design.md（目前未详述I/H）
   和主README；不要机械地把第一轮较高成绩当稳定提升。
9. LaTeX report.tex已经预留memory_design_results.tex，日期改11月2日，PDF尚是前次版本。
   最终跑`bash scripts/build_report.sh`，检查警告并用pdftoppm+view_image看新增页面。
10. 重跑root tests和oGMemory当前HEAD回归（旧9排除已有baseline证明，不隐瞒）。
    提交主仓main和oGMemory m3_dev_1031，不push；最终记录清洁状态、hash和报告路径。
11. 必做工作完成后更新goal complete并向用户报告；若时间/额度限制导致可选组未运行，
    明确写未运行，不制造65题成绩。不要在截止后继续启动模型请求。

## 解释边界

- 本轮是固定记忆QA/表示/检索，不是重新运行AV感知的端到端实验。
- FI只是有来源的名字/角色软排序，不是核实的人脸—声音—姓名绑定；
  初评尚无收益，不能假装身份问题已解决。
- FactIndex为只读原文事实旁路（SQL原记忆不变），不是LLM高层语义记忆或生产规模ANN。
- S相对FIP同时改变事实召回和可见事实no-gain；H又加场景锚点、原生收尾、
  增量补充准入、当前查询身份匹配，均非纯单因素。
- 所有I/H代码在`initial_case_inventory.json`具体参考答案案例检查之前已冻结；
  不要现在据某道题金标写特例。案例诊断已经发现有些错题其实看到了正确词，
  但错误绑定人物或物品；也有近似答案被不同批评委打不同分。最终依同批评分。
- 当前新组不能据初评分推荐替换balanced；待全套结果看质量/用量取舍。
- 5个开发视频、65题、同模型自动0/0.5/1评委，不是人工准确率或官方M3权重复现。
