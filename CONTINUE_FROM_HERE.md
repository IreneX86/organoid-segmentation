# Mito 当前继续点（2026-10-08）

**当前结论：冻结 baseline 保持不变；没有运行中的 GPU 实验。** 最新的单因素 focal BCE+soft-Dice 五折验证策略在 36 张已标注开发图像上的 OOF Dice 为 0.855128，较 baseline 0.854309 仅 +0.000819；低前景 Dice/precision 降低，FP 和远距 FP 增加，因此**reject**。完整数值和安全图表见 [`reports/mito_focal_dice_5fold_20261008.md`](reports/mito_focal_dice_5fold_20261008.md)。frozen tag `baseline-mito-fold0-20261001` 未修改。

- 五折模型、历史、阈值、split、决策、metrics、入选 fold 3/4 的预测及日志已归档在 Git 忽略目录 `runpod_results/mito_focal_dice_20261008/`。五折、最终报告与两份入选预测共八份归档均经 SHA256 核对；五份 checkpoint 非空且 ZIP 可读，历史无 NaN/Inf。
- 唯一 Runpod Secure Cloud A40 Pod `1u5f3be2vobzqn` 已在下载后 TERMINATE/DELETE；`list_pods=[]`、`list_endpoints=[]`、`list_network_volumes=[]`，可见资源当前支出速率 **US$0/hour**。账单末小时仍在入账：已列账 US$0.505487，预计本轮最终约 US$0.95–1.00，稍后以结算明细更新。
- 已完成只读系统误差与数据/标注编码审计，报告在 [`reports/mito_error_and_annotation_audit_20261008.md`](reports/mito_error_and_annotation_audit_20261008.md)。真正的生物学标注语义仍未由独立专家判定；Git 忽略的盲法复核包在 `outputs/mito_annotation_review_20261008/reviewer/`，含 63 个匿名裁剪和 8 张整图。
- **下次从盲法专家复核继续**：两位熟悉 Mito 染色与标注 SOP 的人员先独立看原图，再按固定方案查看 GT。不要改原始 36 对 mask，不用已看过的 OOF 差图定向刷分。复核前不继续跑 ResUNet 或扫描 loss/sampling 参数。未来未见 organoid 数据尚未加入。

## 历史：hard-negative 实验完成记录（2026-10-03）

**状态：本轮实验已完成并归档。** 不要把本轮结果解读为 hard-negative 的泛化增益；frozen baseline tag `baseline-mito-fold0-20261001` 未修改，Cetus 未修改。

- 按 [`experiments/mito_hard_negative_protocol.md`](experiments/mito_hard_negative_protocol.md) 完成 Mito 五折训练，epochs 分别为 76、82、48、57、46。训练前各折只使用自身 training 图像/人工 mask 构建 hard-negative 候选中心，未修改 36 对原始 TIFF。
- 五折 validation-only selection decision **已经锁定，0/5 折入选**。因此没有对 hard-negative 候选读取或评分任何本折 outer-test 标签；门控后的 36-image OOF 完全复用 frozen baseline。结果见 [`reports/mito_hard_negative_5fold_20261002.md`](reports/mito_hard_negative_5fold_20261002.md) 与 `experiments/results/mito_hard_negative_*`。
- 验证集专用 FP/recall 诊断**已经完成**。没有未完成的 GPU 诊断；这些后验诊断没有改变锁定 decision。必要的小型表格已进入 `experiments/results/`。
- 五个模型/checkpoint、split、history、threshold、metrics、GPU 监测及日志归档和最终 decision/OOF 归档保存在本地 Git 忽略目录 `runpod_results/mito_hard_negative_20261002/`。六个关键归档逐一经 SHA256 和内容检查，五个 `.keras` 均非空，训练及诊断指标均为有限值。不要把此目录提交到 Git。
- 首次 preflight 在训练前发现 **72/72** 个 `*.tif`，原因是 macOS tar 同时解包出 36 对 `._*.tif` AppleDouble 元数据。失败日志已保存；在**同一 Pod** 中建立仅过滤这些元数据的 `Organoid_clean` 副本，核对为真实 **36/36** 对后，保持全部科学配置不变重试。原始 TIFF、人工 mask 和失败日志未改动。下次打包应避免 AppleDouble，不能把它们当成生物图像。
- 唯一 Runpod Secure Cloud A40 Pod `txecsmz46nxbtl` **已 TERMINATE/DELETE**。2026-10-03 复核 `list_pods=[]`、`list_endpoints=[]`、`list_network_volumes=[]`，当前可见资源支出速率 **US$0/hour**。本次单价 US$0.49/hour，预算上限 US$2。Runpod 对该 Pod 完整运行时段的结算记录为 **US$0.9671261566691101**（GPU US$0.9607604146003723、disk US$0.006365742068737745）。

**当时结论：**hard-negative 被 reject，冻结 baseline 保持不变；未入选候选折的 raw outer-test 表现没有测量，不能从门控后的零差值推断其分数。
