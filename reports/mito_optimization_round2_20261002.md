# Mito 2D segmentation｜冻结基线与两轮单因素优化归档

**归档日期：**2026-10-02

**研究范围：**36 对已人工标注的独立 2D Mito TIFF／mask；真正待预测的外部 organoid 图像尚未加入。
**当前决策：**本轮没有找到可靠超过 frozen baseline 的方案，默认模型、训练方法和五折划分保持不变。

## 1. 研究问题与比较基准

问题是：在不改人工 mask、不使用 outer-test 标签做模型选择的条件下，能否提高 36 张已标注图像的五折 out-of-fold（OOF）Dice，特别是前景面积较小图像的表现？这批图像属于**开发数据池**；每折的 test 只是内部 held-out 数据，不能代表尚未加入的外部未见图像。

冻结基线是 2D U-Net、512×512 原像素裁剪、前景中心 patch 概率 0.60、BCE+soft-Dice、原有增强和 Adam 1e-4。验证／推理使用重叠滑窗，将概率拼回原始 1200×1200 坐标。最多 100 epochs，EarlyStopping 与 checkpoint 均按该折 validation 选择；阈值也只由 validation 决定。五折划分固定，36 张图各恰好作为一次 outer test。原 fold-0 baseline tag 为 `baseline-mito-fold0-20261001`，未移动。

| 冻结基线 OOF 指标（逐图，n=36） | Mean ± sample SD | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| Dice | 0.854309 ± 0.062776 | 0.875561 | 0.726555 | 0.929643 |
| IoU | 0.750677 ± 0.094024 | 0.7787 | 0.570543 | 0.868536 |
| Precision | 0.839055 ± 0.099596 | 0.8852 | 0.640627 | 0.955861 |
| Recall | 0.877258 ± 0.051503 | 0.8691 | 0.736024 | 0.971350 |

各折 baseline mean Dice：fold 0 **0.869262**、fold 1 **0.860361**、fold 2 **0.857867**、fold 3 **0.822490**、fold 4 **0.860659**。更完整的 split、逐图指标及阈值比较见 [`baseline/mito_five_fold_20261001/`](../baseline/mito_five_fold_20261001/README.md)。

## 2. 已完成实验与结果解释

所有候选均保持同一 frozen split，一次主要改变一个因素。候选训练、checkpoint、阈值与逐折选择只能使用该折的 training／validation。先保存全部五折选择决策，随后仅对被选中的候选折读取 outer-test 标签；未选中折直接复用冻结基线结果。因此下表的“门控后 OOF”是**预先定义的选择策略**成绩，不能当作被拒绝候选模型的 raw 36-image 测试成绩。

| 单因素实验 | 验证选择规则与结果 | 门控后 OOF Dice | 决策 |
| --- | --- | ---: | --- |
| 前景中心 patch 概率 0.60→0.35 | 原预注册规则要求最佳 validation Dice 增加 ≥0.010；0/5 折入选 | 0.854309，等于 baseline | Reject；验证增益小且方向不一致 |
| BCE+soft-Dice → BCE+Tversky，FP/FN 权重 0.6/0.4 | 相同 +0.010 规则；0/5 折入选 | 0.854309，等于 baseline | Reject；五折没有稳定验证增益 |
| 四次→三次下采样 U-Net | 新探索性规则同时约束平均、≤5% 前景子集、单图最差变化和验证方差；仅 fold 3 入选 | **0.854928 ± 0.062555** | 不提升为 current best；收益集中在一折 |
| 推理 tile overlap 128→256 | 同一已冻结 checkpoint、阈值、512 patch；五折 validation 改变约 −0.000040 至 +0.000068；0/5 入选 | 0.854309，等于 baseline | Reject；效果可忽略且无需重训 |

三次下采样候选仅移除最深一层及相应解码层，参数量约从 7.85M 降到 1.95M；归一化、loss、采样、增强和训练预算没有变化。五折候选均完成，但只有 fold 3 在验证规则下被选中。门控策略的逐图配对平均 Dice 变化为 **+0.000619**；mean IoU **0.751588**，precision **0.837998**，recall **0.879598**。36 张中 6 张提高、1 张下降、29 张因复用 baseline 而不变；最差 Dice 仍为 **0.726555**。低前景子集平均 Dice 从 **0.791388** 到 **0.791909**，增量仅 **+0.000522**。fold 3 的策略均值从 **0.822490** 到 **0.825671**，其他折不变。这不足以称为跨折稳定改善；另外四折的**原始三层模型 outer-test 成绩没有测量**。

采样与 Tversky 使用较早预注册的 +0.010 规则；三层 U-Net／overlap 使用另一个事先固定、兼顾平均与尾部的探索性规则。两套规则不能视为统计检验，也不能拿门槛差异作候选优劣比较。详细逐折决策与逐图结果见 [`experiments/results/`](../experiments/results/)；实验假设及规则分别见 [`experiments/mito_sampling_035_protocol.md`](../experiments/mito_sampling_035_protocol.md)、[`experiments/mito_tversky_06_protocol.md`](../experiments/mito_tversky_06_protocol.md)、[`experiments/mito_three_pool_protocol.md`](../experiments/mito_three_pool_protocol.md)、[`experiments/mito_overlap256_protocol.md`](../experiments/mito_overlap256_protocol.md)。

## 3. 低前景图像与错误线索

人工 mask 中前景占比 ≤5% 的图像有 **7/36** 张：baseline mean Dice **0.791388**，mean precision 约 **0.727**；其余 29 张分别约 **0.870** 与 **0.866**。两组 recall 均约 **0.877**。先前只读 OOF 审查中，两组每个背景像素的 FP 率都约 **1.1%**；当前可支持的解释是，相近的背景误报量在前景更少时对 precision 与 Dice 的影响更大（基数效应）。这**不证明**误报的图像学成因相同，也不排除特定背景纹理或边界错误。

约 74% 的 FP 像素属于也与真实前景相交的预测连通区域，因此“仅删除孤立小组件”缺乏直接支持。先前观察到内部 tile 接缝附近 FP 率较高，但 overlap 256 的五折验证几乎没有变化，这不支持仅增加滑窗重叠。原始 TIFF 的高强度分位数跨度较大，现有逐图 1–99.8 百分位归一化已消去部分量纲差异；目前**没有证据**说明替换归一化一定有效。边界扩张、局部亮背景、细长分支、fragmentation／merging、对象尺度与标注一致性尚需以多图的 GT／预测／概率叠图细查，不能提前当作已证实结论。

## 4. 方法学限制与过拟合风险

- 五折指标来自同一个仅 36 张的已标注开发池。内部 test 标签没有参与该折 checkpoint、early stopping、阈值或候选选择，但研究者已看过 baseline OOF 错误，后续假设属于**探索性**，这些旧 folds 不是全新的外部验证。
- 初始 fold map 利用全部 36 张 mask 的前景比例进行分层；原 organoid／Z 身份无法可靠恢复，不能排除相关图像跨折。此限制应随所有内部 OOF 指标披露。
- 只有一折被选中的微小 OOF 增益不足以证明更好泛化。不同 folds 的 epoch／阈值和低前景尾部表现提示不稳定性，但单凭 patch-training loss 与全图 validation Dice 也不能直接断言严重过拟合。
- 未测量的 raw 候选 outer-test 分数不得用门控后 baseline 分数代替。未来新增独立标注数据才可能提供更有说服力的泛化评估。

## 5. 计算资源、费用与结果保管

三个优化阶段均使用单张 Runpod Secure Cloud A40（标价 US$0.49/hour），Python 3.11、TensorFlow 2.21.0、NumPy 2.4.6、tifffile 2026.3.3；每次训练前完成 GPU preflight。采样实验及一次超时的 Tversky pilot 账单 **US$1.373148**；完整 Tversky 五折 Pod `a83ubzuyu2ypi8` 的已结算账单 **US$0.960396**；三层 U-Net 与 overlap Pod `y8ifdpcq0r06oe` 的已结算账单 **US$1.029376**。这三阶段合计 **US$3.362920**；不含更早的 frozen baseline 训练费用。每阶段先下载并核对归档／checkpoint，再删除 Pod；最后一次核查 `list_pods` 为空。大模型、预测 TIFF、GPU 日志及传输归档仅在本地被 Git 忽略的 `runpod_results/` 保存，不进入仓库。没有改变 Cetus 作业。

## 6. 当前结论与下一步

**Keep：**冻结四层 2D U-Net baseline、原归一化、原训练参数和验证选阈值策略。三层 U-Net 的代码保留为可复现实验选项，但不作为默认或 production 候选。**Reject：**此次测试的 sampling 0.35、BCE+Tversky 0.6/0.4、overlap 256；不围绕其参数做密集扫描。

下一步先做**不训练的系统误差分析**：在多个低／高前景与低／高 Dice 样本上同时检查原图、GT、预测、可获得的概率、FP/FN，以及误报相对真实边界和图像亮度的位置；统计背景强度、前景形态和预测连通性。明确区分测量结果与可能的标注差异。只有这些证据指向某种可操作机制，才预先定义一个单因素候选与逐折 validation 选择规则；在此之前不创建新的 GPU 资源。
