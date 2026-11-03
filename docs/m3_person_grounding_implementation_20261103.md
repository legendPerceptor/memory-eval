# 单视频人物关联实现记录

实现提交：`c237bfef22e4474cd3d16c85d9182b15f2f7d613`，分支`m3_dev_1031`，未push。
详细接口与限制见 `oGMemory/docs/m3_person_grounding.md`。

## 已实现

- SQL/内存存储在查询阶段按视频过滤；独立聚类入口必须给出单视频。
- 旧人脸缺失video/clip来源时，经帧和原始视频节点补齐，缓存检测不重复调用CV。
- 完整管线在情景生成前建立按时间顺序的视觉身份候选；不回填未来姓名或最终簇映射。
- 当前片段人脸裁剪和匿名ID进入模型；只接受实际输入ID，保存人物/事件的来源关联。
- 姓名关联仍标为model_claim，要求逐字转写证据；同名及冲突姓名不做人物扩展。
- QA可按来源支持的姓名声明或已观察到的匿名ID召回关联事件，与原检索软融合。
- 来源与锚点hash校验、时间边界、缓存失效、图谱引用幂等更新和摘要去重保护。

## 验证

- 相关单元及内存/专用SQL集成测试：**1094 passed**。
- 根仓评测工具：**109 passed**。新增模块及测试ruff通过。
- 真实study_18只读验证：账号共24083个人脸，仅加载目标视频4467个；
  4422个有效观测，匹配约11.13秒。
  552个前缀观测的身份赋值在加入后续观测后保持一致。
- 以上只读检查不写实验库、不调用CV/聊天模型。专用SQL写入测试只使用独立测试库。

## 未验证与保留边界

新匹配是有限样本的因果原型，不是原全量单链接算法的等价优化。
保守策略产生630个非歧义轨迹候选、2139个歧义观测；
可能过度拆分。候选数不是人数，不能据此宣称身份识别或QA质量提升。
新方案额外输入至多8张人物裁剪，需要独立生成记忆、重建索引并评测。
没有自动做脸—声—姓名认证；未验证的关联不作为硬过滤。

正在运行的十视频仍使用原冻结源码与配置，指纹已核对未变。因此本次提交不会中途加速旧队列，
也不会把原有成绩改写成新方案成绩。用户的LaTeX未提交修改保持不动。

## 只读复查命令

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 oGMemory/.venv/bin/python scripts/smoke_m3_person_identity.py \
  --config experiments/m3_ten_fullinterval_20261102/private/config.yaml \
  --account m3-ten-fullinterval-20261102 --video study_18 \
  --work-dir runs/m3_ten_fullinterval_20261102/study_18 \
  --output runs/m3_person_identity_dev_20261103/readonly_smoke.json
```
