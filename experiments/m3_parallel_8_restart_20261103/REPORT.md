# oGMemory 并发十视频复测

仅重跑oGMemory；M3推理及评分沿用冻结结果，不是新的同步对照实验。
MiniMax-M3.1-Flash / adaptive / medium；统一0/0.5/1评分口径。
失败计零，评委错误另列；均分不是人工准确率。

| 视频 | 题数 | 旧oGMemory | 新oGMemory | 冻结M3 |
|---|---:|---:|---:|---:|
| study_09 | 12 | 0.5833 | 0.5000 | 0.5000 |
| kitchen_01 | 13 | 0.3846 | 0.3077 | 0.1923 |
| bedroom_05 | 12 | 0.4167 | 0.4167 | 0.3750 |
| study_23 | 14 | 0.6786 | 0.5714 | 0.3214 |
| kitchen_23 | 12 | 0.5417 | 0.5000 | 0.6250 |
| study_05 | 12 | 0.4583 | 0.5417 | 0.3750 |
| gym_03 | 11 | 0.7273 | 0.5909 | 0.7273 |
| bedroom_04 | 11 | 0.8182 | 0.9091 | 0.4545 |
| study_18 | 14 | 0.4286 | 0.2500 | 0.0000 |
| kitchen_20 | 17 | 0.0000 | 0.6765 | 0.0000 |

## 资源比较

| 题集/阶段 | 旧/新 | prompt tokens | completion tokens | 未知usage请求 | 墙钟秒 |
|---|---|---:|---:|---:|---:|
| all_128/memory_and_index | old_ogmemory | 3196070 | 989928 | 0 | 49794.0 |
| all_128/memory_and_index | new_ogmemory | 3978504 | 1603812 | 0 | 18114.8 |
| all_128/qa | old_ogmemory | 4634177 | 126469 | 0 | 4355.5 |
| all_128/qa | new_ogmemory | 5315040 | 146394 | 3 | 1988.4 |
| matched_first_eight_97/memory_and_index | old_ogmemory | 2510078 | 784774 | 0 | 37348.5 |
| matched_first_eight_97/memory_and_index | new_ogmemory | 3126005 | 1242828 | 0 | 16847.4 |
| matched_first_eight_97/qa | old_ogmemory | 4039490 | 111694 | 0 | 3954.8 |
| matched_first_eight_97/qa | new_ogmemory | 3998225 | 110918 | 3 | 1725.4 |

## 限制

旧M3 study_18存在连接失败，kitchen_20缺失；旧oGMemory kitchen_20失败。
全128题与前八视频97题的资源指标分开记录。不能把失败/缺失预测的少量用量解释为效率优势。
总时间包含配额等待；不能将不同运行窗口的墙钟差全部归因于并发。
新版本包含人物关系链，不是纯并发消融；失败和未知usage不当作零成本。
并发是否减少token由实测决定，不预设结论。
