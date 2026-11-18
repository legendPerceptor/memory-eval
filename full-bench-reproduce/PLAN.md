# M3-Bench 全量并行复现计划（本机 t4-omni）

本目录按同事的 `runbook.md` 复现 **全量 100 视频 baseline**（禁用 6 个阶段，
即 runbook §4.2 `run_maas.sh` 的 flag 集合），并适配到本机（无 `ogmem-m3runner`
容器、使用本机自部署 embedding / CV 服务、LLM 走华为 MaaS）。

参考基线（同事同配置 full-100）：**0.514 weighted mean，1269/1276 答对，
2 video workers 下 ~25h wall（ingest≈16h，QA≈6h，judge≈14min）**。

> 注意：本机把同事的 `Qwen3-Embedding-0.6B`（:18000）换成了本机 GPU 上的
> `BAAI/bge-m3`（:8001）。两者都是 1024 维，但**模型不同，分数不能视为严格
> 同配置复现**，只能作近似对照。其余（LLM 角色、pipeline、stage 开关、
> QA/judge 参数）与 runbook 保持一致。

---

## 1. runbook 与本机的差异映射

| 项 | runbook（同事机器） | 本机（t4-omni） | 处理 |
|---|---|---|---|
| runner | `ogmem-m3runner` 容器 | 无容器，直接用 `oGMemory/.venv/bin/python` | `run_full_bench.sh` 直接跑宿主机 |
| Postgres/pgvector | `rychkov-ogmem-postgres` :15432 | 容器 `m3p4-pg-20261113` :31877 | 脚本内默认 PG_HOST/PORT/CONTAINER，密码取自 `runs/.m3p4_pgpass_20261113` |
| Embedding | Qwen3-Embedding-0.6B :18000/1024 | 本机 GPU TEI `BAAI/bge-m3` :8001/1024 | YAML `embedding.model/base_url`，dim 保持 1024 |
| CV | insightface :18081 | 同（容器 `t4-omni-cv-gpu-1`） | YAML `cv_service.base_url` |
| M3 数据 | `/data/m3_data` 挂到 `/data_m3` | `/data/m3_data`（100 视频/1276 题） | `ANNOTATIONS` / `VIDEOS_ROOT` 直指宿主机路径 |
| work 目录 | `/data/m3_work_rychkov` | `/data/m3_runs_yuanjian/$RUN` | `M3_WORK_ROOT` |
| Extraction/Judge LLM | `deepseek-v4.1-flash` | `deepseek-v4.1-flash-1017`（用户指定） | YAML `episodic_llm` / `judge_llm` |
| QA LLM | `glm-5.3` | `glm-5.3`（同 runbook） | YAML `qa_llm` |
| LLM endpoint/key | MaaS | 同 `https://api.modelarts-maas.com/v2` | 环境变量 `LLM_*`；key 见 `runbook.md §2` / `llm.sh` |

---

## 2. 配置文件

- `ogmemory.m3-ds.yaml` — 标准配置（`request_timeout_seconds: 300`）。
  - LLM 角色：extraction=`deepseek-v4.1-flash-1017`，qa=`glm-5.3`，judge=`deepseek-v4.1-flash-1017`
  - embedding=`BAAI/bge-m3` @ `http://127.0.0.1:8001/v1`，dim 1024，batch 8
  - cv=`http://127.0.0.1:18081`
  - `m3.vector_index.backend: pgvector`；storage/vector DSN 走 `${M3_STORAGE_DSN}`/`${M3_VECTOR_DSN}`
  - 所有 secret / per-run 值用 `${VAR}` 占位，由 `run_full_bench.sh` export 后再解析
    （`m3/config.py::_expand_env_ref` 只展开整串 `${VAR}`）。
- `ogmemory.m3-ds600.yaml` — 同上，全部 `request_timeout_seconds` 改 600，用于 QA 顽固超时续跑。

---

## 3. 脚本与命令

- `run_full_bench.sh` — 单次全量/子集运行（阻塞）。等价 runbook §4.2 + §5 启动。
  固定 flag：`--seconds 0`；禁用 6 阶段
  `speaker_diarization speaker_embedding voice_segmentation voice_clustering
  face_clustering semantic_memory_consolidator`；`extract_frames/episodic max_frames=4`、
  `no_audio=true`（ASR 仍运行）；`--qa-top-k 8`、`transcript_top_k=20`、`resume=true`、
  `max_iterations=6`；`--judge`。默认 `VIDEO_WORKERS=2`、`STAGE_WORKERS=4`、`QA_TIMEOUT=300`。
- `check_complete.py` — 续跑终止门（每视频每题都有非空、无 error 的最新回答才 exit 0）。
- `full_run_loop.sh` — 循环调用 `run_full_bench.sh` 直到 `check_complete.py` 通过（runbook §5.1 等价）。

### 3.1 单视频 smoke（先验证端到端）
```bash
cd /home/yuanjian/Development/memory-eval/full-bench-reproduce
RUN=smoke-kitchen08 VIDEOS=kitchen_08 DB=ogmem_smoke_kitchen08 ./run_full_bench.sh
```
产物：`/data/m3_runs_yuanjian/smoke-kitchen08/kitchen_08/{pipeline.log,m3_qa_results.jsonl}`、
`.../_aggregate/m3_eval_summary.json`。

### 3.2 全量并行（detached）
```bash
cd /home/yuanjian/Development/memory-eval/full-bench-reproduce
RUN=full100-ds DB=ogmem_full100_ds \
  nohup ./full_run_loop.sh >/data/m3_runs_yuanjian/full100-ds-launch.log 2>&1 &
```
监控：
```bash
tail -f /data/m3_runs_yuanjian/full100-ds/loop.log
tail -f /data/m3_runs_yuanjian/full100-ds/<vid>/pipeline.log
../oGMemory/.venv/bin/python check_complete.py \
  /data/m3_data/annotations/robot.json /data/m3_runs_yuanjian/full100-ds -v
```

### 3.3 QA 顽固超时续跑（600s 配置，只补不完整视频）
```bash
M3_CONFIG="$PWD/ogmemory.m3-ds600.yaml" M3_QA_TIMEOUT=600 \
RUN=full100-ds ./full_run_loop.sh
```

---

## 4. 并行策略

- **video 级并行 = 2**：runbook 明确 glm-5.3 QA 有 ~10 req/min 上限，故 2 video workers。
- **stage 级并行 = 4**：`OGMEM_M3_MAX_WORKERS=4`（MaaS 限制）。
- 本机 4×T4：embedding 用 GPU1、CV 用 GPU2；GPU0/3 空闲。ASR 默认 CPU。
  后续如需要可用 GPU ASR 提速，但不改动 baseline 口径。

---

## 5. 执行阶段（顺序）

1. **Preflight**（已在 `run_full_bench.sh` 内置）：检查 python / 配置 / annotations /
   PG 容器 / embedding :8001 / CV :18081；创建 fresh DB 并 `CREATE EXTENSION vector`。
2. **Smoke**：单视频（kitchen_08，9 题）端到端，确认 ingest→QA→judge 全链路通、产物齐全。
3. **Full run**：`full_run_loop.sh` detached 跑全部 100 视频。
4. **Resume passes**：loop 内自动续跑；顽固超时按 §3.3 用 600s 配置补跑。
5. **Aggregate**：`/data/m3_runs_yuanjian/full100-ds/_aggregate/m3_eval_summary.json`
   （question-level mean）+ 合并 JSONL。

---

## 6. 已知风险 / 注意事项

- **MaaS 限流**：实测 `deepseek-v4.1-flash-1017` 约 **4 req/s**，并发压测会触发
  `ModelArts.81101: rate limit is 4 per second`。按用户要求**先按 runbook 的
  2 video workers / 4 stage workers 跑**，遇到限流再谈退避/降并发。
- **embedding 非同配置**：bge-m3 ≠ runbook 的 Qwen3-Embedding-0.6B，分数仅近似可比。
- **fresh DB**：M3 节点 ID 是确定性的、非 account-scoped，必须每个实验新 DB；断点续跑复用同一 DB/RUN。
- **旧镜像 judge 语义**：本机非容器镜像，judge 由 driver `--judge` 负责；resume 后如需
  对某视频重判，可单独跑 `python -m m3.eval`（见 runbook §5）。
- `run_full_bench.sh` 内有 MaaS key fallback（方便本机直跑）；对外分享前建议改为仅读环境变量。

---

## 8. B 方向架构（本机 4×T4，embedding + ASR 都用 GPU）

用户最终决定：**不做 embedding→CPU 的降级方案，embedding 留 GPU，ASR 也上 GPU**
（4 卡：GPU0/GPU3 空闲，正好给 ASR 轮转；GPU1=embedding、GPU2=CV 不变）。

### 8.1 GPU 分配
| GPU | 用途 | 说明 |
|-----|------|------|
| GPU0 | ASR（faster-whisper） | 并发视频 A |
| GPU3 | ASR（faster-whisper） | 并发视频 B |
| GPU1 | embedding（TEI bge-m3） | 常驻 :8001 |
| GPU2 | CV（insightface） | 常驻 :18081 |

driver 以 `--asr-gpus 0,3` 启一个 GPU 池：每个并发视频子进程拿到**独占**的一张卡
（写入 `CUDA_VISIBLE_DEVICES`），视频结束归还。`--video-workers` 默认 = GPU 数（=2）。

### 8.2 ASR 上 GPU 的两个前提（已在本仓库改好）
1. **CUDA 库加载路径**：faster-whisper 的 cuBLAS/cuDNN 在 venv 的 nvidia wheels 里，
   不在默认 loader path。`run_full_bench.sh` 已 export：
   ```
   LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/{cublas,cudnn,cuda_nvrtc}/lib
   ```
   否则报 `libcublas.so.12 is not found`。
2. **compute_type**：T4 用 `float16`（无 bf16）。原先 `voice_asr` 只能传 `--device`，
   默认 `int8`；已给 `m3/voice_asr.py` 加 `--compute-type`，`pipeline/stages.py` 转发，
   `run_full_bench.sh` 传 `--set voice_asr.params.device=cuda` +
   `--set voice_asr.params.compute_type=float16`。

实测：GPU `cuda/float16` 约比 CPU `int8` 快 **~10.7×**（RTF 0.010 vs 0.105）。

### 8.3 embedding 稳定性加固
上次 TEI 因框架级队列 bug 崩溃（Rust `core/src/queue.rs:87` panic → Exit 139），
且原 `restart:'no'` 导致容器永久停、loop 空转。已加固：
- `deploy-llm-on-gpu/compose.yaml`：embeddings 段 `restart: unless-stopped` + `/health` healthcheck。
- 对**正在运行**的容器已 `docker update --restart unless-stopped`（无需重建即生效）。
- `full_run_loop.sh` 每轮 preflight 前 `ensure_services`，发现 embedding/CV 掉线自动 `docker start`。

### 8.4 全量运行命令（本机）
```bash
cd /home/yuanjian/Development/memory-eval/full-bench-reproduce
# 续跑（复用已完成 9 视频的 DB ogmem_full100_ds；MAX_PASSES=0 不限轮次）
setsid bash -c 'RUN=full100-ds DB=ogmem_full100_ds MAX_PASSES=0 exec ./full_run_loop.sh' \
  >/data/m3_runs_yuanjian/full100-ds-launch.log 2>&1 </dev/null &
```
注意：必须用 `setsid ... &`，普通 `nohup &` 会被 exec 会话回收。

### 8.5 监控
```bash
tail -f /data/m3_runs_yuanjian/full100-ds/loop.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
cd full-bench-reproduce && ../oGMemory/.venv/bin/python check_complete.py \
  /data/m3_data/annotations/robot.json /data/m3_runs_yuanjian/full100-ds
```
