# PR #343 CI 诊断与修复

## 失败原因

核对 PR 的公开页面和其链接的 Jenkins 日志：

- PR 源提交：`c237bfef22e4474cd3d16c85d9182b15f2f7d613`。
- 父任务 `oG-memory #1613` → 子任务 `oG-memory_compile #1611`。
- CI 先将 PR 合并到 `dev`，然后执行：

```bash
pip3 install -e '.[dev,opengauss]'
coverage run -m pytest tests/unit
```

失败发生在测试收集阶段，而不是某条测试的断言：

```text
collected 8553 items / 1 error / 2 skipped
ERROR collecting tests/unit/m3/test_person_memory.py
tests/unit/m3/test_person_memory.py:9: from PIL import Image
ModuleNotFoundError: No module named 'PIL'
```

pytest 返回 2，流水线因此停止。此时没有完成全量 UT，覆盖率输出也不能用来
判定正常运行后的覆盖率水平。

`PIL` 由 Pillow 分发包提供。新人物记忆 UT 直接导入它，实际
`m3.person_memory.build_person_context` 的人脸裁剪同样直接依赖它；
项目此前没有声明这项直接依赖。本地多模态实验环境已间接安装 Pillow，
所以此前 1,094 项相关测试没有暴露干净安装环境中的缺依赖问题。

父任务中另有标签更新请求返回 401/405，属于 CI 机器人接口/凭据问题，
不是本次子任务 pytest 中断的根因。本修复不修改 CI 机器人凭据。

## 修复

分支 `m3_dev_1031` 新增提交：

```text
1cd9378795c39b23959e6b2ad86eae9a783ac890
fix(m3): declare Pillow runtime dependency for person grounding
```

仅两个文件、五行新增：

- `pyproject.toml`：将 `pillow>=10.0` 声明为直接运行依赖；
- `uv.lock`：同步主项目依赖与 requires-dist，保留已有 Pillow 锁定版本，
  未批量升级其他依赖。

不是通过 `importorskip` 隐藏新增测试，也不是只向 CI 临时安装。
不修改模型、提示、检索、人脸算法、评测数据或正在运行的冻结实验。
并发分支 `intergation_1103` 保持 `eceb96ff` 不变。

## 复现与验证

- 新建无系统包继承的 Python 3.11.17 虚拟环境，仅按 CI extras 安装。
- 修复前，仅收集 `test_person_memory.py` 就复现相同的 `PIL` 缺失、
  pytest 退出码 2。
- 修复后重新安装同一 extras：元数据直接包含 Pillow，
  人物记忆模块 **28 passed**。
- `uv lock --check --offline` 和 `git diff --check` 通过。
- 全量验证使用独立工作区，组合 `dev@b1156b26`、PR 源 `c237bfef` 与上述
  两文件修复，以复现 CI 的合并方式，而不是只测试源分支。

直接在宿主机试跑全量 UT 时，部署用例假设 18090 空闲，与本机现有服务冲突。
该次运行主动中止，已执行部分为 9 failed / 2326 passed / 5 skipped，
**不是全量结果，也不归因于本 PR**。没有停止本机服务或修改部署测试。

随后用独立 user/network namespace、启用 loopback，执行完整 CI 测试命令：

```bash
cd runs/pr343_ci_20261103/merged
unshare -Urn bash -c '
  ip link set lo up
  export PYTHONPATH="$PWD"
  /home/yuanjian/Development/memory-eval/runs/pr343_ci_20261103/venv/bin/python \
    -m coverage run -m pytest tests/unit
'
```

隔离环境正常收集 **8,581 条测试，另有 2 项收集期跳过**，
原先未能收集的 28 条人物记忆测试已包含在内。
本地 Python/pytest 等补丁版本与 CI 不完全一致，远端最终结论仍需实际重新运行 CI。

完整运行已结束，命令退出码 **0**：

```text
8531 passed, 52 skipped, 14 warnings in 501.93s (0:08:21)
```

没有使用 `-k`、`--ignore` 等选项排除失败用例，也没有新增 skip；
52 项按用例/收集时既有条件跳过，不计为通过。14 条汇总警告为既有
`pytest.mark.unit` 未注册警告；另有启动时 asyncio 默认 fixture scope 提醒。
原先受宿主端口影响的部署测试在隔离环境中全部通过。
本地的完整 UT 命令通过不代表远端流水线已重跑通过。

## 留存文件

日志位于 Git 忽略的 `runs/pr343_ci_20261103/`：

- `parent-1613.log` / `compile-1611.log`：下载的 CI 日志；
- `install-before.log` / `reproduce-before.log`：干净环境安装及失败复现；
- `install-after.log` / `lock.log`：依赖修复及锁文件同步；
- `person-after.log` / `lock-check.log`：28 条人物 UT 及锁文件检查；
- `unit-after.log`：宿主机首次运行及端口冲突；
- `merge.log` / `pillow.patch`：CI 合并与修复输入；
- `ci-merged-unit.log`：隔离网络环境中的完整 UT 日志。

## 更新 PR

目前只做本地提交，未 push，未触发远端重跑。上传此修复到已有 PR 的源分支：

```bash
git -C oGMemory push yuanjian m3_dev_1031:m3_dev_1031
```

并发分支未包含本依赖修复；后续整合时应带上该提交，或在新的
`m3_dev_1031` 基础上重新整理并发提交。不要修改正在运行的冻结实验源码。
