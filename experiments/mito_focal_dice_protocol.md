# Mito 单因素 focal BCE + soft-Dice 探索方案

**状态：2026-10-08 在运行前固定。** 本方案只研究已有 36 对人工标注的 2D Mito 图像。冻结 baseline tag、五折 map、原始 TIFF/mask 和所有先前实验结果不变。由于研究者已经看过这些图的 baseline OOF 错误，本轮仍是**探索性开发实验**，不能作为独立外部泛化证明。

## 假设与局限

冻结 baseline 的低前景（≤5%）图像 precision 明显较低，而 recall 与其他图相近；远离 GT 的亮结构误报也更常见。一个信息量高的单因素问题是：让现有损失更加关注**按现行 mask 定义的难分类像素**，是否能降低误报而保持 recall？这不预设那些亮结构在生物学上一定不是线粒体。独立盲法标注语义复核另见 `experiments/mito_annotation_semantics_review_protocol.md`，尚不能把未知标注语义解释为已证实标签错误。

## 唯一改变的训练因素

将 baseline `mean(BCE) + (1 − per-image soft Dice)` 中的 BCE 项替换为**无 class-alpha、γ=2 的对称 binary focal BCE**：`mean[(1 − p_t)^2 × BCE(y,p)] + (1 − per-image soft Dice)`，其中 `p_t = y p + (1−y)(1−p)`，概率剪裁 epsilon 只用于对数稳定性。focal 项系数固定为 1，不调整 gamma、class weight 或 Dice 项。保留四层 2D U-Net、0.60 前景中心 patch 采样、无 hard-negative 分支、1–99.8% 逐图归一化、512 patch、batch 4、100 steps/epoch、最多 100 epochs、Adam 1e-4、原增强、ReduceLROnPlateau、EarlyStopping、128 overlap 和 validation 选阈值。LD/Nuc 不变。不与 Tversky、three-pool 或先前采样候选组合。

## 五折与选择隔离

使用已经冻结的五折 map；每折训练仅打开 train/validation 标签，使用 validation 做 checkpoint、early stopping、阈值和与 frozen baseline 的**同图配对比较**。先训练全部五折并把 validation-only 决策一次性锁定，再只对入选折读取 outer-test mask；未入选折的 OOF 行复用 frozen baseline，不测其原始候选 test Dice。完全复用 `mito_round2_gate.py` 的事前固定门控：每折平均 validation Dice Δ≥+0.003，或低前景子集 Δ≥+0.010 且总体非负；同时低前景 Δ≥−0.005、单图最差 Δ≥−0.020、validation Dice SD 增量≤+0.015。门槛是实用保护线，不是显著性检验。

锁定后报告 36 图 selected-policy OOF 的配对 Dice/IoU/precision/recall、低前景、最差图和各折一致性；特别检查 precision 增益是否以 recall 损失换来。**不要**因单折结果不好而重训，也不要为提高 outer-test Dice 修改本协议。仅 validation gate 的入选并不自动晋升 production baseline；若跨折不稳定或生物标注语义未澄清，只保留为探索性候选。

## 运行与费用边界

只允许一台已验证的 Runpod Secure Cloud A40，创建前读取实时单价和账户余额；本轮新增费用上限 US$5。训练前做 CUDA/TensorFlow 单 GPU preflight，输出目录独立；每折完成即归档并校验本地 checkpoint、history、split、threshold 和日志。结果安全下载后立即删除 Pod，确认无运行资源和 US$0/hour。原图、mask、checkpoint、预测 TIFF、Runpod 日志与凭据不提交 Git。
