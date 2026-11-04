# m3_dev_1031 提交说明与证据索引

## 直接使用

- PR 标题与完整正文：`docs/PR_m3_dev_1031.md`。
- 可附在 PR 的 token 对比图：`docs/figures/m3_dev_1031_pr_tokens.png`。
- 原始指标摘录、派生比例、来源文件 SHA-256：`docs/m3_dev_1031_pr_evidence.json`。
- 重建证据与图片：`oGMemory/.venv/bin/python scripts/build_m3_pr_evidence.py`。

正文中的仓库内文档路径相对于 oGMemory；实验原始数据在本评测仓。
提交时将图作为 PR 附件上传，不直接复制一个目标仓内不存在的本地图片路径。
本次仅准备材料，**没有创建远端 PR、没有 push、没有改动 m3_dev_1031 提交历史**。

## 分支核对（2026-11-03）

- 源分支：`m3_dev_1031`，`c237bfef22e4474cd3d16c85d9182b15f2f7d613`。
- 目标建议：`opengauss/oGMemory:dev`；已重新 fetch。
- 当前目标：`b1156b26`；共同基点：`8c86c3ed`。
- 源分支独有 23 个提交；目标分支在共同基点之后有 4 个提交。
- PR 三点 diff：52 个文件，4,954 行新增、99 行删除。
- PR #300 并发整合 `eceb96ff` **不属于本 PR**。

机器 Git 不支持新版 `merge-tree --write-tree`，因此使用隔离的 detached worktree
执行了 `git merge --no-commit --no-ff origin/dev`，成功无冲突后立即 abort。
这不改变任何正式分支，也不等于合并后完整测试通过。

在该隔离工作区、合并之前，对精确的源提交重新运行测试：
`runs/m3_dev_1031_pr_review_20261103/tests.log`：**1094 passed in 13.79s**。
测试脚本为同目录 `run_tests.py`，SQL 写入只使用独立测试数据库。

需要上传时，从评测仓执行（由提交者确认远端权限后手工运行）：

```bash
git -C oGMemory push yuanjian m3_dev_1031:m3_dev_1031
```

此命令明确上传 m3_dev_1031，即使当前 checkout 的是 intergation_1103，
也不会上传并发分支。随后选择源 `hython/oG-Memory:m3_dev_1031`、
目标 `opengauss/oGMemory:dev`，粘贴 `PR_m3_dev_1031.md`。
目标分支后续若又变化，应重新检查合并与 CI。

## 最适合作为 PR 主证据的结果

1. 固定记忆上下文消融：输入 19,905,749 → 4,804,362（−75.9%），
   有效回答 59/65 → 65/65；保留 balanced 独立重复，不只展示较高分。
2. 同期 balanced → 严格增量 E：输入 −38.2%、未缓存输入 −63.6%，
   65/65 有效回答；自动均分 0.5308 → 0.5846，只作为探索性观察。
3. 30 秒采样覆盖修复：原 0–15 秒前缀 → 约 0–28 秒均匀基础帧，
   不宣称单独产生某个提分幅度。
4. 真实人脸只读 smoke：账号 24,083 条 → 当前视频 4,467 条；
   这是处理范围改善，不是直接的 81.5% 时间下降。

PR 内的 A/B 表来自不同共同评委批次。同一固定答案在不同批次可能得到不同评分，
不得把批次一的高分与批次二的低基线拼成更大的增益。

## 十视频结果为什么不作为“完整 PR 已提分”的主标题

最终十视频使用 `8d61b772`，包含抽帧与 E 方案，但不包含最新 `c237bfef` 人物关联。
结果为 finished_with_failures：

- oGMemory 109/128 有效、自动均分 0.48046875；
- M3 100/128 有效、自动均分 0.33203125；
- oGMemory `kitchen_01` 有两题首次超时；`kitchen_20` 在 embedding 阶段失败，
  17 题缺失；
- M3 `study_18` 有 11 条连接错误、`kitchen_20` 未执行。

若补充展示，可用最终同一批共同评分中的前八视频固定 97 题：

| 视频 | 题数 | oGMemory | M3 同模型参考适配 |
|---|---:|---:|---:|
| study_09 | 12 | 0.5833 | 0.5000 |
| kitchen_01 | 13 | 0.3846 | 0.1923 |
| bedroom_05 | 12 | 0.4167 | 0.3750 |
| study_23 | 14 | 0.6786 | 0.3214 |
| kitchen_23 | 12 | 0.5417 | 0.6250 |
| study_05 | 12 | 0.4583 | 0.3750 |
| gym_03 | 11 | 0.7273 | 0.7273 |
| bedroom_04 | 11 | 0.8182 | 0.4545 |
| 题目加权合计 | 97 | 0.5722 | 0.4381 |

这不是重新挑选答对题：保留八视频全部 97 题，oGMemory 两题首次超时仍计零
（95/97 有效），M3 为 97/97 有效。该子集用于避开末两视频的未完成阶段，
不是预登记的独立验证集，也不能代替公开全 128 题结果。

上述数值必须从**最终十视频共同评分**重新汇总；不要混入早期八视频单独评委
得到的 0.5361 / 0.4381。两题后来重跑的诊断评分均为零，且未覆盖主结果。

八视频 QA 输入为 oGMemory 4,039,490、M3 733,810，说明仍有优化空间，
不支持“比 M3 更省 token”。此前人脸阶段累计 309.29 分钟反映旧的全账号重算，
不能与新人物方案约 11 秒的单视频只读匹配直接相除，声称等价端到端加速。

## 原始证据位置

| 证据 | 路径 |
|---|---|
| 上下文消融及共同评分 | `experiments/m3_context_study_20261101/REPORT.md`、`comparison.json` |
| 来源事实/严格增量及共同评分 | `experiments/m3_memory_design_20261102/REPORT.md`、`comparison.json` |
| 增量预算和前缀审计 | `experiments/m3_memory_design_20261102/strict_delivery_audit.json` |
| 最终十视频评分 | `experiments/m3_ten_fullinterval_20261102/joint_questions.jsonl` |
| 单片段故障注入 | `experiments/m3_study18_recovery_20261101/partial-fault-injection.json` |
| 人物方案真实数据只读验证 | `docs/m3_person_grounding_implementation_20261103.json` |

人物方案的有限样本因果匹配与旧全量单链接算法不等价，候选数也不是人数。
当前并发复测包含人物方案与并发多项变化，即使完成，也不能自动作为人物机制
单独贡献或该 PR 纯粹加速的证据。
