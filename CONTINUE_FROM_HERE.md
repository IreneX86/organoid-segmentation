# Mito hard-negative 实验完成记录（2026-10-03）

**状态：本轮实验已完成并归档。** 不要把本轮结果解读为 hard-negative 的泛化增益；frozen baseline tag `baseline-mito-fold0-20261001` 未修改，Cetus 未修改。

- 按 [`experiments/mito_hard_negative_protocol.md`](experiments/mito_hard_negative_protocol.md) 完成 Mito 五折训练，epochs 分别为 76、82、48、57、46。训练前各折只使用自身 training 图像/人工 mask 构建 hard-negative 候选中心，未修改 36 对原始 TIFF。
- 五折 validation-only selection decision **已经锁定，0/5 折入选**。因此没有对 hard-negative 候选读取或评分任何本折 outer-test 标签；门控后的 36-image OOF 完全复用 frozen baseline。结果见 [`reports/mito_hard_negative_5fold_20261002.md`](reports/mito_hard_negative_5fold_20261002.md) 与 `experiments/results/mito_hard_negative_*`。
- 验证集专用 FP/recall 诊断**已经完成**。没有未完成的 GPU 诊断；这些后验诊断没有改变锁定 decision。必要的小型表格已进入 `experiments/results/`。
- 五个模型/checkpoint、split、history、threshold、metrics、GPU 监测及日志归档和最终 decision/OOF 归档保存在本地 Git 忽略目录 `runpod_results/mito_hard_negative_20261002/`。六个关键归档逐一经 SHA256 和内容检查，五个 `.keras` 均非空，训练及诊断指标均为有限值。不要把此目录提交到 Git。
- 首次 preflight 在训练前发现 **72/72** 个 `*.tif`，原因是 macOS tar 同时解包出 36 对 `._*.tif` AppleDouble 元数据。失败日志已保存；在**同一 Pod** 中建立仅过滤这些元数据的 `Organoid_clean` 副本，核对为真实 **36/36** 对后，保持全部科学配置不变重试。原始 TIFF、人工 mask 和失败日志未改动。下次打包应避免 AppleDouble，不能把它们当成生物图像。
- 唯一 Runpod Secure Cloud A40 Pod `txecsmz46nxbtl` **已 TERMINATE/DELETE**。2026-10-03 复核 `list_pods=[]`、`list_endpoints=[]`、`list_network_volumes=[]`，当前可见资源支出速率 **US$0/hour**。本次单价 US$0.49/hour，预算上限 US$2。Runpod 对该 Pod 完整运行时段的结算记录为 **US$0.9671261566691101**（GPU US$0.9607604146003723、disk US$0.006365742068737745）。

**下一阶段起点：**本轮无待完成的训练、下载、诊断或账单核对。实验结论为 **reject hard-negative，保持 frozen baseline**。若考虑下一轮，先安排分层、盲法的标注语义复核；新训练需单独定义实验与预算，不能根据未测量的 raw candidate outer-test 分数推断效果。
