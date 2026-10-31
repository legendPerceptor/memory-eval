# LaTeX 报告构建

## 一键构建

在工作区根目录执行：

```bash
bash scripts/build_report.sh
```

也可以执行 `make -C latex`，或者从任意目录用构建脚本的绝对路径调用。

- 成品：`latex/report.pdf`
- 编译输出：`latex/build/build.log`
- TeX 详细日志：`latex/build/report.log`
- PDF 元数据：`latex/build/pdfinfo.txt`
- 源文件及 PDF 哈希：`latex/build/build-manifest.txt`

脚本使用 XeLaTeX 和 latexmk，禁用 shell escape，并自动完成所需的多轮编译。
每次强制重新检查源文件，以免 `results.tex` 从不存在变为存在后仍沿用旧的阶段报告。
构建通过文件锁串行执行；成功后才原子替换成品，失败时保留此前 PDF。
它不会启动模型实验、修改预测或生成虚构成绩。

## 当前文档内容

`report.tex` 包含当前方案、实验协议、限制和 M3-Agent 对照章节。

- 有 `results.tex` 时，纳入实际完整运行生成的数值章节。
- 没有该文件时，封面标记“阶段报告”，展示明确注明日期的临时成绩。
- `m3_agent_comparison.tex` 是参考实现的源码级比较，不代表已跑过 M3-Agent。
- `findings.tex` 的案例审计明确标注所覆盖的视频和临时评分。

`scripts/finalize_m3.py` 已接入构建脚本：主驱动结束、汇总及审计成功后，
重新生成报告内容并编译 PDF。其状态文件中的 `reports_generated` 才代表
该轮自动收尾已成功；`needs_attention` 表示需要检查日志。

## 本机补齐的依赖

系统已有基础 TeX、latexmk、xetex 二进制及 PDF 检查工具，
但初次检查缺少 xelatex 命令、ctex/xeCJK 和 Fandol。
本次没有 sudo 权限，也没有修改系统包或 Python/GPU 环境。
从现有 apt 源下载 `texlive-xetex`、`texlive-lang-chinese`、
`texlive-lang-cjk` 的发行版包，仅将必要 TeX 文件补充到用户目录：

- TeX 文件和字体：`~/texmf`
- XeLaTeX 命令链接：`~/.local/bin/xelatex`
- 用户格式文件：由 `fmtutil-user` 创建于用户 TEXMFVAR
- 下载缓存与格式构建日志：`~/.cache/memory-eval-tex`

类似环境可显式运行：

```bash
bash scripts/setup_latex_user.sh
```

该脚本是基础 TeX 已安装时的补充安装，不是完整 TeX 管理器；
保留已存在的用户文件，不自动升级系统组件。系统 TeX 更新后，
用户级 format 可能覆盖新的系统格式，需要重新运行相应的 `fmtutil-user` 构建。
不要直接删除整个 `~/texmf`，其中也可能有用户其他文档的依赖。

## 实际验证

本次已成功构建 8 页阶段报告，执行了：

1. 从工作区外的 `/tmp` 调用构建脚本。
2. 用模拟的 latexmk 失败验证非零退出码，确认原 PDF 哈希不变。
3. 恢复正常工具后重新编译。
4. 用 pdfinfo 和 pdftotext 检查页数、中文标题、阶段标记、对照章节及临时分数。
5. 渲染并查看封面和对照表页面；没有 overfull box 或 missing-character 警告。

当前仍有 Fandol 字体 OpenType CJK script 标签的非致命 fontspec 提醒；
没有把“编译成功”描述成“完全无警告”。验证快照见
`docs/latex_build_validation.json`，工具链记录见 `docs/latex_toolchain.json`。
