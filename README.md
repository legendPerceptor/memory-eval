# oGMemory 记忆改进与评测工作区

本目录用于固定实验配置、保存可审计结果，以及撰写中文技术报告。
主要代码仓为 `oGMemory/`；部署仓为 `deploy-llm-on-gpu/`；
`locomo-test/` 保留用于后续文本记忆评测，本轮没有新增 LoCoMo 实验。

## 最新实验：来源事实、人物条件与增量交付（2026-11-02）

**715条交错共同评分已完成，11条完整65题轨迹，零评委错误。**
新跑了7条oGMemory轨迹（454/455有效）和1条M3参考QA重复（65/65有效），
同时对前次B两条固定答案及M3参考固定答案共同重评。视频仍是原来的5个，
这是固定记忆的QA/表示/检索实验，不是重新生成AV感知。

| 方案 | 本批共同评分 | QA输入token |
|---|---:|---:|
| 前次B两条固定答案 | 0.5923 / 0.6000 | 4.804M / 4.708M |
| 本轮同期B | 0.5308 | 4.721M |
| S：来源事实索引，两次 | 0.5846 / 0.6077 | 3.457M / 3.409M |
| E：严格3500增量+收尾请求，一次 | 0.5846 | 2.916M |
| M3参考适配，原轨迹/本轮重复 | 0.4615 / 0.4615 | 0.570M / 0.552M |

S均值0.5962与前次B两条答案的本批均值相同；E较同期B减少38.2%总输入、
63.6%未缓存输入，但只运行一次，**不能宣称稳定全面提分**。
F/FI/FIP等低分消融全部保留，见
`experiments/m3_memory_design_20261102/REPORT.md`和`comparison.json`。
方法与边界：`docs/m3_memory_design.md`；
分析结论：`docs/m3_memory_design_findings.md`。
通用只读事实索引构建入口为`python -m m3.build_fact_index`，
说明见`oGMemory/docs/m3_source_fact_index.md`。新功能均默认关闭。

**用量采集限制：**所有QA用量记录完整；共同评委的启动采集因源码路径缺失失败，
714个需模型评分任务的usage未持久化。评分保留且可校验，没有重评成功项或估算补账；
不能给出完整的整轮总token/费用。问题和后续启动预检修复均已记录。

## 前一轮上下文实验（2026-11-01，历史结果保留）

**上下文优化及455条交错共同评分已完成。** 本批共同评分：
原固定oGMemory答案59/65、0.5385；M3参考文本适配65/65、0.4615；
A（仅预算）65/65、0.5538；B（预算+收尾）65/65、0.6077；
B独立重复65/65、0.5769；C（12k预算）64/65、0.5231；
D（分组原文）65/65、0.5385。
B两次均分的描述性平均为0.5923，不择高、不拼接答案。
原固定答案此前0.5462与本批0.5385的差异来自重评，预测没有修改。
协议、状态及用量/错误审查见 `experiments/m3_context_study_20261101/`；
设计说明见 `docs/m3_context_optimization.md`。

| 实验 | 模型 | 样本 | 状态 |
|---|---|---|---|
| `m3_baseline_20261031` | Omni-7B | 固定 5 视频 / 65 题 | 按用户要求中断，不能作为完整质量基线 |
| `m3_minimax_flash_20261031` | MiniMax-M3.1-Flash / max | 相同固定样本 | 首片段推理耗尽，已停止，保留作诊断 |
| `m3_flash_medium_20261031` | MiniMax-M3.1-Flash / medium | 相同 5 视频 / 65 题 | 已结束，报告已生成；请核对 REPORT.md 中的失败与完整性检查 |
| `m3_study18_recovery_20261101` | 相同模型与预算 | 定向恢复 study_18 / 14 题 | 84/84 记忆、14/14 有效答案，均分 0.3929；保留首轮结果 |
| `m3_agent_matched_20261101` | M3-Agent-style + 相同 MiniMax | 相同 5 视频 / 65 题 | 已完成新感知和 340 条记忆；两种控制器及统一评分见最终对照报告 |

报告源码：`latex/report.tex`。方案梳理、实验计划、配置、结果与审计记录：
`experiments/m3_flash_medium_20261031/`。
不要将报告中的“运行中”版本引用为最终实验成绩。
本轮已安排 `scripts/finalize_m3.py` 等待主驱动退出，随后执行固定分母汇总、
只读审计并生成 `REPORT.md`、`latex/results.tex`，随后构建 `latex/report.pdf`；监控状态见
`experiments/m3_flash_medium_20261031/finalization_status.json`。
如果主进程被中断、没有完整 run report，收尾会标记 `needs_attention`，
不会把中断结果伪装成完成。收尾脚本不重跑模型或择优修改答案。

## 复现方式

环境已安装在 `oGMemory/.venv/`。使用现有环境，不自动更新 GPU 依赖。
用户 `~/.bashrc` 有非交互 shell 提前返回逻辑，因而用交互 bash 加载：

```bash
# 配置已经存在，不要在同一实验上重复 prepare。
# 仅首次配置（会创建专用数据库并启用 vector；发现非空目标库会拒绝）：
bash -ic 'source ~/.bashrc; export MINIMAX_API_KEY; oGMemory/.venv/bin/python scripts/prepare_m3_flash.py --experiment-id m3_flash_medium_20261031 --database ogmem_m3_flash_medium_20261031 --account m3-flash-medium-20261031 --reasoning-effort medium'

# 检查固定样本，不调用模型：
bash -ic 'source ~/.bashrc; export MINIMAX_API_KEY; bash scripts/run_m3_flash_medium.sh --selection-only'

# 正式运行；已有完成视频会跳过，但失败视频按原 wrapper 重建证据：
bash -ic 'source ~/.bashrc; export MINIMAX_API_KEY; bash scripts/run_m3_flash_medium.sh'

# 运行结束后固定分母汇总与只读审计：
python3 scripts/summarize_m3.py \
  --run runs/m3_flash_medium_20261031 \
  --output experiments/m3_flash_medium_20261031
oGMemory/.venv/bin/python scripts/audit_m3_artifacts.py \
  --run runs/m3_flash_medium_20261031 \
  --experiment experiments/m3_flash_medium_20261031
```

**不要在运行中的实验上再次启动上述命令。** 原 wrapper 不锁定运行配置，
所以改模型、预算、采样或检索策略应新建实验 ID、输出目录及隔离数据库，
不能通过原地修改私有 YAML 把不同条件混在同一成绩里。
本轮模型密钥只通过环境变量读取，GPU 服务密钥留在权限为 600 的私有配置中。

## 记录约定

- `experiments/<id>/PLAN.md`：开始前固定协议，后续变化明示。
- `provenance.json`、`source.patch`：源码提交、兼容修改和哈希。
- `config.redacted.yaml`：脱敏配置；真实配置在 `private/`，不提交。
- `sample_manifest.json`：候选集合、种子、固定视频、题数与标注哈希。
- `input_inventory.json`：实际视频路径、ffprobe 时长和媒体参数。
- `questions.csv` / `questions.jsonl`：固定全集逐题答案、分数、理由与失败标记。
- `metrics.json`：题目加权、视频宏平均、覆盖率与错误统计。
- `model-usage.jsonl`：逻辑请求元数据，无提示、媒体、密钥。
- `database_audit.json` / `artifact_audit.json`：图与向量覆盖、账号隔离、
  来源文件哈希及观察到的阶段时长。
- `runs/`：完整媒体、管线日志和原始结果，体积较大，默认不入 Git。

同模型自动评分只作为诊断，不等同于独立人工准确率。
缺题和模型失败必须计入固定全集；评委故障不能当作回答错误。
失败恢复应保存初次结果，不择优拼接多轮预测。

## 测试与报告

```bash
oGMemory/.venv/bin/python -m pytest scripts/test_*.py -q
cd oGMemory
.venv/bin/python -m pytest tests/unit/m3 tests/unit/providers/llm -q
cd ../latex
make
# 或在任意目录执行：
bash /home/yuanjian/Development/memory-eval/scripts/build_report.sh
```

构建入口为 `scripts/build_report.sh`，成品是 `latex/report.pdf`，
编译日志与输入哈希位于 `latex/build/`。脚本使用 XeLaTeX + latexmk，
支持从任意目录运行；只有编译成功才原子替换已发布的 PDF。
当前 PDF 为不超过8页（含参考文献）的五章论文：Introduction、Related Work、Method、
Evaluation、Conclusion and Discussion。正文统一使用11月2日最终共同评分，
比较同模型M3-Agent参考适配；历史过程、逐题结果与工程审计仍保留在 `docs/`、
`experiments/` 和旧的LaTeX片段中，不再全部插入正文。构建依赖 `pdfinfo`，
编译后检查页数；超出8页时失败并保留原已发布PDF。
本机补充的中文 TeX 文件位于 `~/texmf`，XeLaTeX 链接位于 `~/.local/bin`；
没有修改系统包、Python/GPU 依赖或使用 sudo。其他同类机器若缺中文支持，
可手动运行 `bash scripts/setup_latex_user.sh`（需要基础 TeX 和 apt 下载能力）。
系统 TeX 升级后需重新生成用户级 XeLaTeX format。

## M3-Agent 对照分析

- `docs/m3_agent_comparison.md`：英文源码级流程梳理、差异矩阵、评测协议与复现前置检查。
- `docs/m3_agent_review.json`：参考提交、标注一致性、权重目录检查；明确未运行参考评测。
- `latex/m3_agent_comparison.tex`：保留的历史详细比较；当前论文的Related Work包含精简分析。

源码审计阶段发现两套原生默认评委、评分规则、时间边界与模型预算不同，
不能直接拿这些原生分数比较；后续适配实验另行统一了模型与评测设置。
参考仓库保持不变，未运行其安装脚本或部署官方模型权重。
另已完成 `comparisons/m3_agent_matched/` 中的同模型适配实验，详见对应实验的
`PLAN.md`、`status.json`、`adapter_source_hashes.json`。它使用冻结源代码和新感知数据，
不能称为官方权重复现。oGMemory 的容错改动已在 `m3_dev_1031` 本地提交，未推送。

各代码目录各自有 Git 历史，外层 `.gitignore` 默认不嵌套提交，
而用实验提交号和补丁记录版本。外层已建立Git仓库，当前工作在main分支。

## 首次同模型对照的最终汇总（2026-11-01）

统一评分且修复评委限流后：oGMemory（恢复后固定预测）0.5462，
M3-Agent 参考 Action 文本协议适配版 0.4615，原生工具迁移诊断版 0.2154。
三组分别完整保留，不能将不同控制器的最佳单题拼接。
详见 `experiments/m3_end_to_end_comparison_20261101/REPORT.md`、`questions.csv`。
这些不是官方 M3-Agent 权重或论文复现分数，也不是严格单变量/墙钟性能结论。

## 使用有界QA配置

当前oGMemory提供 `python -m m3.configure_qa_context`，生成新的0600私有配置副本，
只调整 `m3.qa_context`，不更换模型或连接。可选择balanced/compact/grouped，
必须提供本地tokenizer；命令与限制见 `oGMemory/docs/m3_bounded_context.md`。
它应用于ReAct QA/pipeline，不是旧的 `python -m m3.qa` 单次入口。
跨策略使用新的输出目录，不能resume旧答案。当前论文式消融固定原记忆，
因此不要为了改QA上下文重新运行全部视频抽取。

## 11月1日上下文实验的结论与限制（历史）

- 质量优先时先采用balanced（22528有效输入估计上限、6轮/180秒）。
  B两次实际输入约4.8M/4.7M，峰值约16.9k，原固定轨迹为19.9M/156.7k。
- 新五组轨迹共325题次，324条有效回答、无QA超时；C的一条协议失败原样计零。
  总分进展主要来自原超时题恢复，原已回答题仍有退步。
- 缓存比例下降，不能把约76%的总输入缩减当作费用缩减。
- 所有五视频整簇重采样区间均包含0；这是开发集、自动评委和同模型适配结果，
  不是官方权重/全数据集复现或独立人工准确率。current_clip也不是严格因果协议。
- 代码与报告只在本地提交，未push。完整逐题共同评分在
  `experiments/m3_context_study_20261101/joint_questions.csv` / `.jsonl`。
