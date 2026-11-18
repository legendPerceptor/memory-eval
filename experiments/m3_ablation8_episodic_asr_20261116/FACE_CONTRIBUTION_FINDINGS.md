# 人脸 / 人物接地对 QA 的贡献：历史证据、机理与 m3-agent 借鉴（Future Work 备查）

创建时间：2026-11-16（Asia/Shanghai）
作者：Codex（基于对历史实验产物与 `m3-agent` 源码的审计）
状态：**只做整理，不启动新实验**。本文是"之后的工作"的设计底稿。

> 阅读顺序：§0 结论 → §1 证据（数字都标了出处/可复现命令）→ §5 机理 → §6 m3-agent 借鉴 → §7 工作建议。

---

## 0. 一句话结论

**现有"人脸识别 + speaker diarization + 人物接地"这套投入，在 oGMemory 主链路上没有可归因的 QA 收益，根因是 name binding 基本没打通（真实姓名绑定率 ≈ 0.15%）。**
因此送进多模态/检索链路的人脸，绝大多数是"外观描述 + 64 位哈希 ID"的**结构性噪音**，不是能让模型"把人脸对上人名"的身份信号。
可控隔离实验（同一记忆图 / 同批题目，仅切换人脸文本）显示：**加了人脸不提升，去掉人脸也不掉**。
`m3-agent` 走的是另一条路（**先用 `<face_k>/<voice_j>` ID 锚定视觉身份，再在一次全局多模态推理里做 face↔voice 等价 + 仅在转写显式出现时抽取姓名**），这是最值得借鉴的思路（§6）。

---

## 1. 证据一：name binding 几乎完全失效（硬数字，已独立复算）

数据源：`runs/m3_dev_1031_ten_retry_20261108/<video>/episodic_memory.jsonl`（论文期含 `person_grounding` 的分支产物，是目前唯一带人脸接地结构的运行）。

统计口径：按 clip 内 `characters[]` 逐条计数；`name_evidence` 以 `person_binding.name_evidence`（或 character 顶层）是否存在为准。

| 范围 | character 条目 | 有 `person_binding` | 有真实 `name_evidence` |
|---|---:|---:|---:|
| paper8 八视频（study_09 / kitchen_01 / bedroom_05 / study_23 / kitchen_23 / study_05 / gym_03 / bedroom_04） | **1325** | **669（50.5%）** | **2（0.15%）** |
| 全部 10 视频（+kitchen_20 / study_18） | 1693 | 892（52.7%） | 5（0.30%） |

那仅有的 2 条真实姓名证据原文：
- `study_23` → `"Rose, please put it back."`
- `gym_03` → `"So Leah, can you provide the back spot for me?"`

也就是说：**ASR 里明明喊出了名字（Rose / Leah），但系统几乎从不把它绑定到对应的 character/face 上。**
"半绑定"（有 person_binding，无 name_evidence）占绝大多数——这类绑定对人名问答毫无用处，因为见 §2。

> 复现命令（见 §8）会打印上表。此数字对交接时"1287 / 677 / 2"做了独立复算修正（口径更严：仅 paper8、按 clip character 计数），结论方向完全一致：**真实姓名绑定率在千分之几量级**。

---

## 2. 证据二：`person_binding` 是"外观 + 哈希"，QA 用不上

典型 character 结构（`study_23` clip0）：

```json
{
  "name": "Woman in light top and long gray skirt",      ← 不是人名，是外观描述
  "role": "Adult who tidies the table and handles the papers",
  "description": "A long-haired woman wearing a light short-sleeved top ...",
  "person_binding": {
    "person_id": "person_5318b4874ec921f842098429",       ← 64 位哈希，无语义
    "face_id": "dc54549b8d7d8a27fa6639dc471f077f",
    "face_content_sha256": "572035392812f28c...",
    "anchor_face_id": "810a463be61cfc7cbc1f3d8a074ba6ac",
    "bbox": [0.675, 0.230, 0.705, 0.311],                 ← 归一化框
    "identity_status": "visual_proposal",
    "association_status": "model_claim",
    "clip_id": 0, "video_id": "study_23", ...
  }
}
```

问题：
- `name` 字段永远填的是外观描述（"Woman in light top..."），**不是名字**；
- `person_id` / `face_id` / `sha256` 是哈希，**模型读了也不知道是谁**；
- `association_status = "model_claim"`、`identity_status = "visual_proposal"` 说明这些"身份"只是模型外观联想，未经验证；
- 真正能让 QA 受益的是"Rose↔这个 face↔这段 voice↔说的这句话"的链条，而这条链在数据里**断在最关键的 name 那一环**。

结论：即便把 person 数据喂给问答模型，它拿到的也只是"某个黑灰连帽衫男人 = person_546497…"，对"谁是 David / Rose 说了什么"没有帮助。

---

## 3. 证据三：person 块把检索文本撑大 1 倍，是噪音不是信号

数据源：`experiments/m3_compact_replay_20261109/offline_index_size.json`（用已存 episodic memory + 转写**重构**的完整索引文本，非 embedding 计费；token 为代理分词器计数）。

| 指标 | 原始（含完整 person 文本） | 精简（compact_person） |
|---|---:|---:|
| 十视频索引总字符 | 4,901,391 | 2,373,574 |
| 压缩幅度 | — | **-51.6%** |
| kitchen_01 每 clip 平均字符 | 6,281 | 3,313 |

- 在 `m3_dev_1031` 那次运行里，`compact_person` **没开**，`index_episodic.py` 把整坨 person JSON（`Person associations (model claims): [...]`）注入检索文本；`kitchen_01` **100% 的 clip（79/79）都带 characters 块**。
- 结果：检索文本一半以上是无人名、纯外观 + 哈希的结构化噪音，**挤占真正有用的叙事与对话**。
- 而 §4 会看到：把这块删掉（compact）**并没有让 QA 变差**，反而原始版还略高一点——说明这块文本既不是主要信号，也不是主要瓶颈，纯粹是**成本**。

---

## 4. 证据四：可控隔离实验——"加了人脸不提升，去掉人脸也不掉"

这是**最干净**的证据：同一记忆图 / 同批题目，只切换"是否使用人脸/人物接地"。

### 4.1 单视频严格消融（study_23，face+person_grounding on vs off）
数据源：`experiments/m3_person_ablation_study23_retry_20261105/judged.json`；manifest：`source_commit=8731930`，design 明确注明 "on enables person grounding and retrieval; off disables both. Not crop-only."

| 臂 | 题数 | 平均分 |
|---|---:|---:|
| off（关人脸/接地） | 14 | **0.5357** |
| on（开人脸/接地） | 14 | **0.5357** |

**完全相同。** 限制：perception 每臂独立重算、非位级一致；每臂仅 1 次（manifest 自带 limitation）。

### 4.2 十视频文本级 A/B（同一图，person 文本完整 vs 精简）
数据源：`experiments/m3_compact_replay_20261109/joint_questions.jsonl`（256 条 = 128 题 × 2 臂，judge_error=0）。

| 臂 | 题数 | 平均分 |
|---|---:|---:|
| original（完整 person 文本） | 128 | **51.17%** |
| compact（精简 person 文本） | 128 | **49.61%** |

完整噪音版**没有更低**（甚至 +1.56pp，属噪声范围）。→ 删掉 person 文本既不伤也没救；它**不承载 QA 信号**。

---

## 5. 证据五：分支对比——大差距是小样本方差，别拿去"给人脸定罪"

数据源：`experiments/m3_three_branches_20261110/summary.json`（3 视频 39 题）与 `experiments/m3_dev_1031_ten_retry_20261108/comparison.json`（10 视频 128 题）。

| 对比 | 视频/题数 | dev（无人脸链条） | m3_dev_1031（人脸+接地） | 其它 |
|---|---|---:|---:|---:|
| three_branches | 3 vids / 39 Q | **58.97%** | 37.18% | simplify 39.74% |
| ten_retry | 10 vids / 128 Q | — | 51.95% | parallel 50.39% |

- 3 视频 39 题里 m3_dev_1031 惨败（kitchen_01 只有 7.69%，系统性把 "Tool box / Material box" 答成 "rolling storage cart"）——但 **39 题方差极大**；
- 扩到 10 视频 128 题后，**m3_dev_1031 与 parallel 基本持平（51.95% vs 50.39%）**。
- 两个对照的模型 / 源码 / 并发不同，**不能把差异单独归因于"人脸"**。

**正确解读**：分支间总分差主要由**小样本方差 + 模型/配置差异**驱动，不是"人脸有用/有害"的证据。真正可信的只有 §4 的两个**隔离实验**，结论是"≈0 贡献"。已有 `experiments/m3_paper8_fresh_20261110/ACCURACY_INVESTIGATION.md` 也得出相同方法学结论：总分差无法归因到单一原因，须做严格消融。

> 参照基线（仅供参考，不用于归因人脸）：`m3_paper8_fresh_20261110` 三臂 = ogmemory_strict 50.52% / paper_og 55.15% / paper_m3 44.85%（97 题；3 题协议失败按 0）；当前 dev 系 `m3_parallel4_paper8_20261113` = 63.92%。

---

## 6. 机理分析：为什么"人脸"送进去反而成了噪音

1. **可检测 ≠ 可接地**：insightface 能稳定检出/聚类人脸（外观向量），但"这是谁"必须靠**名字**。数据里名字来自 ASR，而 ASR 里的名字**没有一条可靠的规则**被回挂到 face 上 → §1 的 0.15%。
2. **半成品结构当信号注入**：把 `person_id`（哈希）、`bbox`、`sha256`、`model_claim` 直接写进检索文本，人读不懂、模型也读不懂 → 纯 token 成本（§3 文本翻倍）。
3. **噪音挤占上下文预算**：检索 text 一半是 person 块，挤掉叙事/对白；但删掉又不掉分（§4.2），说明它本就**不在信号路径上**，只在**抢预算**。
4. **没有 face↔voice 等价**：人名往往在"谁在说话"里出现（diarization 的 voice 段），但 oGMemory 没有把 voice 段和 face 绑成同一 character 的机制，名字和脸**永远对不上**。这是 name binding 打不通的结构性原因。

---

## 7. m3-agent 怎么做 name binding（可借鉴）

源码：`m3-agent/mmagent/`（`face_processing.py` / `src/face_clustering.py` / `memory_processing.py` / `videograph.py` / `retrieve.py` / `prompts.py`）。这是 ByteDance M3-Agent 的开源实现，做法与 oGMemory 很不一样。

### 7.1 视觉身份"先锚定成 ID"，而不是先求名字
- 人脸：insightface `buffalo_l`（RetinaFace 检测 + ArcFace 512 维向量）。
- 聚类：`cluster_faces(faces, 20, 0.5)` → **HDBSCAN**（预计算余弦距离 `1-sim`），只在"好脸"（`detection_score>=0.8 且 quality>=20`）上跑，`min_cluster_size=20`；坏脸进 `-1` 丢弃。
- 跨 clip 续接：新 cluster 的向量去 `search_img_nodes`（embedding 相似）**匹配已有 `img` 节点**，命中就 `update_node` 追加，没命中就 `add_img_node` → **同一个人在整个视频里保持同一个 `<face_k>` 节点**（`max_faces_per_character=3` 只留质量最高的 3 帧）。
- 这一层解决的是"**跨片段人脸连续性**"——oGMemory 的 per-clip hash `person_id` 恰恰没有这个连续性。

### 7.2 一次全局多模态推理里同时做三件事（核心）
`generate_video_context()` 把 **整段视频 + 带框的人脸图（键名 `<face_k>`）+ 按说话人聚好的语音段（键名 `<voice_j>`）** 拼成一个多模态输入，喂 `gemini-1.5`，用 `prompt_generate_full_memory` 要求同时产出 episodic + semantic，且**强制**：

1. **只用 ID 指代角色**：`"If a character has a provided feature ID, refer to them only with the ID (e.g. <face_1>, <voice_2>)"`；**禁止** he/she/they、**禁止**编造名字。
2. **face↔voice 等价**：`"Equivalence: <face_x>, <voice_y>"` —— 把"这张脸"和"这个声音（说话人）"绑成同一人。**这正是 diarization 名字能落到脸上的桥梁。**
3. **名字只在显式出现时给**：`"Name (if explicitly stated)"`，示例输出形如 `"<face_1>'s name is David."`。

### 7.3 等价合并 → character 节点 → 检索时膨胀
`videograph.refresh_equivalences()`：收集所有 `Equivalence:` 语义节点，用**并查集**把 `face_x / voice_y` 归并成 `character_N`，得到 `character_mappings: character_N -> [face_x, voice_y, ...]` 与反向映射。
检索 `retrieve.translate()` 把记忆里的 `<face_x>` 统一替换成 `character_N`；`back_translate()` 再把查询里的 `character_N` 展开成所有 `face/voice` 变体 → **图结构层面的身份扩展，而不是把哈希灌进文本**。

### 7.4 与 oGMemory 的关键差异（为什么值得借鉴）

| 维度 | oGMemory `m3_dev_1031` | m3-agent |
|---|---|---|
| 身份单元 | per-clip 哈希 `person_id` | 跨 clip 持久的 `<face_k>` 图节点 |
| 名字来源 | 要求 `name_evidence` 为**逐字引用**，几乎不命中（0.15%） | 全局推理里"**若显式说出就写** `"<face_1>'s name is David."`" |
| face↔voice | **没有等价机制** | `Equivalence: <face_x>, <voice_y>` 显式绑定 |
| 身份落地方式 | 结构化 JSON **灌进检索文本** | 图节点 + `character_N` 映射，**检索期展开** |
| 检索文本污染 | 严重（person 块撑大一倍） | 无（人名/ID 是图属性，不是原始 JSON） |

**一句话：m3-agent 把"身份"做成了先在视觉上锚定 ID、再用一次全局推理把"脸-声-名"缝起来，并把身份留在图结构里；oGMemory 则在每个 clip 里求一个求不到的 "name_evidence 引用"，求不到就把哈希 JSON 灌进文本当"身份"。**

### 7.5 借鉴时必须诚实说明的共同局限
m3-agent 的名字同样**只在"显式说出"时才抽**。如果某个视频里从头到尾没人喊名字，它也**无法凭空绑定人名**——这是任务本身的硬约束，不是实现缺陷。但即便如此，m3-agent 至少：(a) 能把"这张脸↔这个声音"绑对，(b) 不把哈希噪音塞进检索文本。这两点就足以解释"为什么它的人脸数据不是纯噪音"。

---

## 8. 工作建议（Future Work，按优先级）

**目标**：让"人脸 / diarization"从噪音变成**可接地的身份信号**，并**能被 QA 真正消费**。

1. **改名 binding 的产出方式**（借鉴 m3-agent 7.2/7.4）
   - 放弃"逐 clip 要求 `name_evidence` 逐字引用"这种近乎必然失败的路径；
   - 改为**一次（或按窗）全局多模态推理**：输入 = 视频帧(带 `<face_k>` 框) + diarization 语音段(`<voice_j>`) + ASR 转写，让模型输出 `Equivalence: <face_x>, <voice_y>` 与"显式名字"句；
   - 强约束：仅显式出现才写名字，禁止编造、禁止代词。
2. **建 face↔voice 等价 + 持久身份节点**（借鉴 7.1/7.3）
   - 用 embedding 匹配把同一个人跨 clip 续到同一节点；
   - 用并查集把 `face/voice` 归并成 `character_N`，维护正反向映射。
3. **身份绝不进检索原始文本，只在检索期展开**（借鉴 7.3）
   - 立即止血：把 `person_binding` 的 `person_id/bbox/sha256` **移出** `index_episodic` 文本（§3 已证它撑大一倍且无信号）；
   - 身份作为图边/属性存在，查询时用 `character_N` 做扩展（`back_translate` 思路）。
4. **独立评估 name-binding 质量，而非只看端到端准确率**
   - 指标：被问答引用的命名实体中，**正确绑定到 face/voice 的比例**；以及"ASR 里出现的人名，有多少被绑上"（当前 0.15% 是分母）。
   - 端到端 97 题方差太大（§5），不足以单独驱动决策。
5. **把该能力作为新特性移植回 `dev` 后再做消融**
   - 在 `dev` / `m3-parallel` 当前**根本没有人脸接地**，在其上"关人脸"是结构性 null（见 `CAPABILITY_GAP.md`）；要测人脸贡献，必须在**打通 name binding 的新分支**上做 `FULL vs FULL-人脸`。

> 与当前 8 视频消融（`PLAN.md`：episodic+ASR 最小臂）的关系：那条实验回答"最小臂能到多少"，**不能**回答"人脸贡献多少"。人脸贡献的正解在本文件 §4（已有隔离实验：≈0）与 §8（打通后重测）。

---

## 9. 可复现命令

```bash
# §1 name binding 统计（paper8 八视频）
cd /home/yuanjian/Development/memory-eval
python3 - <<'PY'
import json,os
base='runs/m3_dev_1031_ten_retry_20261108'
paper8=['study_09','kitchen_01','bedroom_05','study_23','kitchen_23','study_05','gym_03','bedroom_04']
tot=bind=nev=0; ex=[]
for v in paper8:
    p=os.path.join(base,v,'episodic_memory.jsonl')
    if not os.path.exists(p): continue
    for line in open(p):
        if not line.strip(): continue
        for c in (json.loads(line).get('characters') or []):
            tot+=1; pb=c.get('person_binding') or {}
            if pb: bind+=1
            ne=pb.get('name_evidence') or c.get('name_evidence')
            if ne: nev+=1; ex.append((v,c.get('name'),ne))
print(f'characters={tot} person_binding={bind} ({100*bind/tot:.2f}%) name_evidence={nev} ({100*nev/tot:.2f}%)')
for e in ex: print(e)
PY

# §4.1 study23 on/off
python3 - <<'PY'
import json; from collections import defaultdict
d=json.load(open('experiments/m3_person_ablation_study23_retry_20261105/judged.json'))
agg=defaultdict(list)
for r in d: agg[r['arm']].append(r['score'])
for a,v in agg.items(): print(a,'n=',len(v),'mean=',round(sum(v)/len(v),4))
PY

# §4.2 compact replay A/B
python3 - <<'PY'
import json; from collections import defaultdict
rows=[json.loads(l) for l in open('experiments/m3_compact_replay_20261109/joint_questions.jsonl') if l.strip()]
agg=defaultdict(list)
for r in rows:
    if r.get('judge_score') is not None: agg[r['arm']].append(r['judge_score'])
for a,v in agg.items(): print(a,'n=',len(v),'mean=',round(sum(v)/len(v),4))
PY

# §3 索引文本体积
cat experiments/m3_compact_replay_20261109/offline_index_size.json
```

---

## 10. 数字出处与限制一览

| 结论 | 数字 | 出处 |
|---|---|---|
| name binding 失效 | paper8: 1325 char / 669 bind / **2 name_evidence（0.15%）** | 本文 §1 复算（`runs/m3_dev_1031_ten_retry_20261108`） |
| 单视频严格消融 | on=off=**53.57%**（14 题各） | `m3_person_ablation_study23_retry_20261105/judged.json` |
| 文本级 A/B | original **51.17%** vs compact **49.61%**（128 题各） | `m3_compact_replay_20261109/joint_questions.jsonl` |
| 索引体积 | 4.90M → 2.37M 字符（**-51.6%**） | `m3_compact_replay_20261109/offline_index_size.json` |
| 分支对比 | dev 58.97% / m3_dev_1031 37.18% / simplify 39.74%（39 题） | `m3_three_branches_20261110/summary.json` |
| 分支对比（大样本） | m3_dev_1031 **51.95%** vs parallel **50.39%**（128 题） | `m3_dev_1031_ten_retry_20261108/comparison.json` |

**限制**：
- §4.1 每臂仅 1 次、perception 非位级一致；§4.2 为文本级复盘（复用同一图，非完整重跑）——但两者方向一致。
- §5 分支对比混入了模型/并发差异，**不可单独归因人脸**。
- §1 的 `characters` 计数随切窗/重跑略有浮动（交接时 1287/677/2，本次复算 1325/669/2）；**结论不在精确值，而在"真实姓名绑定率在千分之几"**。
- m3-agent 借鉴部分为**源码阅读结论**，尚未在本仓 robot 数据集上实测其 name-binding 命中率（列为 §8 待办）。
