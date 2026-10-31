# oGMemory 记忆改进与评测工作区

本目录用于固定实验配置、保存可审计结果，以及撰写中文技术报告。
主要代码仓为 `oGMemory/`；部署仓为 `deploy-llm-on-gpu/`；
`locomo-test/` 保留用于后续文本记忆评测，本轮没有新增 LoCoMo 实验。

## 当前实验

| 实验 | 模型 | 样本 | 状态 |
|---|---|---|---|
| `m3_baseline_20261031` | Omni-7B | 固定 5 视频 / 65 题 | 按用户要求中断，不能作为完整质量基线 |
| `m3_minimax_flash_20261031` | MiniMax-M3.1-Flash / max | 相同固定样本 | 首片段推理耗尽，已停止，保留作诊断 |
| `m3_flash_medium_20261031` | MiniMax-M3.1-Flash / medium | 相同 5 视频 / 65 题 | 已结束，报告已生成；请核对 REPORT.md 中的失败与完整性检查 |

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
python3 -m unittest discover -s scripts -p 'test_*.py' -v
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
当前 PDF 明确标记为阶段报告，最终结果出现后会重新构建。
本机补充的中文 TeX 文件位于 `~/texmf`，XeLaTeX 链接位于 `~/.local/bin`；
没有修改系统包、Python/GPU 依赖或使用 sudo。其他同类机器若缺中文支持，
可手动运行 `bash scripts/setup_latex_user.sh`（需要基础 TeX 和 apt 下载能力）。
系统 TeX 升级后需重新生成用户级 XeLaTeX format。

## M3-Agent 对照分析

- `docs/m3_agent_comparison.md`：英文源码级流程梳理、差异矩阵、评测协议与复现前置检查。
- `docs/m3_agent_review.json`：参考提交、标注一致性、权重目录检查；明确未运行参考评测。
- `latex/m3_agent_comparison.tex`：已纳入中文技术报告的比较章节。

当前两个系统的评委、评分规则、时间边界与模型预算不同，不能直接比较现有分数。
参考仓库保持不变，未运行其安装脚本或占用现有 GPU 部署参考模型。

各代码目录各自有 Git 历史，外层 `.gitignore` 默认不嵌套提交，
而用实验提交号和补丁记录版本。用户可以之后把外层目录初始化为 Git 仓库。
