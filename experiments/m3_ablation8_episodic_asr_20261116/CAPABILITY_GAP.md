# 能力/接线核对：`dev` vs `rychkov/m3-parallel`（更正版）

日期：2026-11-16　作者：Codex
**本文件更正 PLAN.md 早期把"论文冻结快照"当作基准所导致的错误结论。**

## 0. 结论先行

1. **`rychkov/m3-parallel` 相对 `dev` 没有删除任何能力。**
   - `dev` 是 `m3-parallel` 的**祖先**：`git merge-base --is-ancestor dev rychkov/m3-parallel` → true。
   - 领先关系：`dev...rychkov/m3-parallel` = `0 156`（dev 已全部包含，parallel = dev + 156 个提交）。
   - `git ls-tree` 全仓对比：**dev 有而 parallel 没有的文件 = 0 个**。
   - 关键文件函数级对比（qa.py / search.py / control_adapter.py / episodic_memory.py / index_episodic.py）：
     **dev 里的每个 `def` 在 parallel 里都还在**（0 缺失）。
   - 那些 numstat 里的"deletions"（如 qa.py -116、extraction.yaml -426）都是**重构/改写**，不是删除功能。
2. **`fact_index.py` / `fact_memory.py` / `person_memory.py` 属于论文期未合入 dev 的实验代码**，
   出自 `m3_dev_1031` / `simplify_person_info_1110` / `intergation_1103` 分支（含提交 `8731930`），
   **`dev` 和 `master` 从来没有这些文件**。所以不是"parallel 删掉了它们"，而是"它们本就从未进主干"。
3. **`same_identity` 边和 `MemoryNode` 不进 QA 检索，是 `dev` 的既有设计，不是 parallel 的回归。**
   - `dev` 里 `same_identity` 的消费者同样只有 `face_clustering` / `voice_clustering` / `wipe_stages`（清理）。
   - `dev` 里 `control_adapter.py` / `search.py` / `index_episodic.py` 同样只消费 `extractor_type=episodic_memory_generator`，不碰 MemoryNode。
4. **`person_grounding` 也不在 `dev`**：`git grep person_grounding/person_memory/person_identities dev` → 空。

## 1. 这意味什么（纠正之前的判断）

- 我此前说"m3-parallel 把 fact_index 整条检索删了"——**错误**。真相是：fact_index 那套是论文期的实验分支成果，
  **从未合入 dev**；parallel 只是"dev + 并行/吞吐等演进"，天然不含它。
- 所以"聚类算子/身份信息对 QA 没贡献"不是 parallel 引入的问题，而是**主分支（dev）当前的产品设计**：
  QA 只做 episodic 文本检索 + 相邻 clip 扩展，未做图边遍历、未索引 MemoryNode、未做身份条件检索。

## 2. 你真正想做的事，正确的落点

- **并行能力**：`m3-parallel` = dev + 156 提交（含并行/并发写入/抽取等增强）。它**可以直接快进合入 dev**
  （dev 是它的祖先，合入 = fast-forward，不丢任何 dev 的东西）。这一步本身安全。
- **让 fact_index / person_grounding / same_identity / MemoryNode 发挥作用**：这些是**主干上从未存在的能力**，
  需要作为**新功能**从论文分支移植/重做，而不是"从 parallel 里抢救回来"。
- 即：**先合 parallel 进 dev 拿并行；再把论文期能力作为独立特性分支移植进 dev，并各自配 QA 检索接线 + 消融验证。**

## 3. 对当前 8 视频消融实验的影响

- 在 `m3-parallel`（≈dev+性能）上做"只留 episodic+ASR"消融：**成立且有意义**，测的是当前主分支设计的真实上限。
- 在 `m3-parallel` 上做"关人脸/关 diarization"消融：会得到**诚实的近似 null**（因为 dev 本就不把身份送进 QA），
  这本身是"当前主分支下这些算子对 QA 无直接贡献"的证据，**不是** parallel 的缺陷。
- 若要回答"人脸/diarization 到底值不值"，正确路径二选一：
  (a) 在**含 person_grounding 的论文分支（m3_dev_1031 / 8731930）**上做 `FULL vs FULL−face`；或
  (b) 先把这些能力作为新特性移植进 dev 并接好 QA 检索，再在同一基线上消融。

## 4. 证据命令（可复现）

```bash
git -C oGMemory merge-base --is-ancestor dev rychkov/m3-parallel   # true
git -C oGMemory rev-list --left-right --count dev...rychkov/m3-parallel   # 0  156
comm -23 <(git -C oGMemory ls-tree -r --name-only dev|sort) \
         <(git -C oGMemory ls-tree -r --name-only rychkov/m3-parallel|sort)  # 空
for f in m3/fact_index.py m3/fact_memory.py m3/person_memory.py; do
  git -C oGMemory cat-file -e dev:$f 2>/dev/null && echo "$f in dev" || echo "$f NOT in dev"; done  # 全部 NOT in dev
git -C oGMemory branch -a --contains 8731930   # m3_dev_1031 / simplify_person_info_1110
```
