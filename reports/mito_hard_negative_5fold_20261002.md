# Mito 亮背景 hard-negative 采样五折实验

**日期：**2026-10-02。**结论：Reject；冻结 baseline 不变。** 本实验只研究 36 对已标注、相互独立的 2D Mito TIFF/mask。未来真正待预测的 organoid 图像尚未加入项目。

## 假设、固定方案与隔离

冻结 baseline 的 36 张 OOF 图像中，前景 ≤5% 的 7 张 mean Dice 为 0.791388、precision 为 0.727167；它们的 FP 绝对数与其他图像接近，部分远离 GT 且局部较亮。这启发了 [事先写定的方案](../experiments/mito_hard_negative_protocol.md)：前景中心 patch 保持 **0.60**；原 0.40 非前景分支分为 **0.20 原 uniform crop + 0.20 亮背景 hard-negative 中心 crop**。中心必须属于该折训练 mask 的背景、距前景 >10 原像素，且该训练图像归一化强度及减去 Gaussian σ=12 px 后的局部残差均达到各自 GT-background 的 p90。每个训练图像单独计算统计与候选中心；没有读取本折 validation 或 outer-test 来构建采样器。五折训练图像各有 4,274–40,045 个合格中心，无回退。

四层 U-Net、BCE+soft-Dice、1–99.8% 逐图归一化、512×512 patch、batch 4、100 steps/epoch、最多 100 epochs、Adam 1e-4、增强、EarlyStopping、128-pixel overlap 与 validation 选阈值均保持 baseline 设置。五折 frozen split 未变。全部候选模型先完成训练；[固定 validation-only gate](../experiments/results/mito_hard_negative_candidate_decisions.json) 保存五折选择后，才允许被选折的 outer-test 导出。**本次 0/5 折被选，因此没有读取任何 hard-negative 候选模型的 outer-test 标签或测得其 raw candidate OOF Dice。**

| Fold | 完成 epochs | 候选减 baseline 的 validation mean Dice | ≤5% 前景 validation Dice 变化 | 选择 |
| --- | ---: | ---: | ---: | --- |
| 0 | 76 | −0.002276 | −0.006308 | 否 |
| 1 | 82 | +0.000833 | +0.002656 | 否 |
| 2 | 48 | −0.000682 | +0.001808 | 否 |
| 3 | 57 | +0.000211 | +0.002541 | 否 |
| 4 | 46 | +0.000383 | −0.023823 | 否 |

各折最佳 validation Dice、中心数量及归档 SHA256 见[实验清单](../experiments/results/mito_hard_negative_manifest.json)。所有折正常结束，训练记录中的 loss/validation Dice 有限；没有因指标差而重跑任何折。

## 锁定策略的五折 OOF 比较

由于 gate 全部拒绝，**门控后的 36-image OOF 方法逐张复用冻结 baseline**。下表是这个锁定选择策略的成绩，**不是未入选 hard-negative 模型的 test 成绩**。

| OOF 指标（逐图，n=36） | Frozen baseline | Hard-negative 门控后策略 | 配对变化 |
| --- | ---: | ---: | ---: |
| Dice mean ± sample SD | 0.854309 ± 0.062776 | 0.854309 ± 0.062776 | 0 |
| Dice median / min / max | 0.875561 / 0.726555 / 0.929643 | 同左 | 0 |
| IoU mean | 0.750677 | 0.750677 | 0 |
| Precision mean | 0.839055 | 0.839055 | 0 |
| Recall mean | 0.877258 | 0.877258 | 0 |
| 低前景 Dice / precision mean（n=7） | 0.791388 / 0.727167 | 同左 | 0 / 0 |
| FP 像素总数 | 533,991 | 533,991 | 0 |
| 距 GT >10 px 的 FP 像素总数 | 108,432 | 108,432 | 0 |

36 张的逐图 OOF 指标及配对差值见[OOF 明细](../experiments/results/mito_hard_negative_candidate_oof_summary.json)；二值预测与人工 mask 坐标、Dice 复核后得到的 FP 明细见[逐图表](../experiments/results/mito_hard_negative_selected_oof_fp_per_image.csv)。全部 36 张配对 Dice 变化均为 0，worst-case Dice 仍为 **0.726555**。这些零变化是事先规定的 gate 回退结果，不能证明原始候选模型的 FP 也不变。

## 只用验证集的误差诊断

在五折决策锁定**之后**，仅用每折 inner validation 图像及各自事先选定的阈值比较两个模型的 FP、距 GT >10 px 的 FP 与 recall。此分析没有改变选择规则，也没有读取本折 outer-test；[逐图配对数据](../experiments/results/mito_hard_negative_validation_fp_paired.csv)仅作诊断。

| Validation 平均每图变化：候选−baseline | 全部 36 张 | 前景 ≤5% 的 7 张 | 其余 29 张 |
| --- | ---: | ---: | ---: |
| Dice | −0.000292 | −0.002682 | +0.000285 |
| Precision | −0.000780 | −0.000231 | −0.000912 |
| Recall | +0.000675 | **−0.005463** | +0.002157 |
| FP 像素 | +119 | −370 | +237 |
| 距 GT >10 px 的 FP 像素 | +115 | −55 | +156 |

低前景组虽然平均每图 FP 少了约 370 个，但 recall 同时下降，Dice 反而下降；fold 4 的低前景 validation Dice 下降 0.023823。全部验证图像有 21 张 Dice 上升、15 张下降，但均值略降；没有跨折稳定的验证收益，也没有稳定减少远距 FP。小样本上的这些诊断差值应理解为描述性结果，不能当作独立泛化证据。

## 运行记录、限制与下一步

使用**一台** Runpod Secure Cloud A40 Pod `txecsmz46nxbtl`，目录独立，标价 US$0.49/hour；Python 3.11.16、TensorFlow 2.21.0、NumPy 2.4.6、tifffile 2026.3.3、SciPy 1.16.2。一次技术性 preflight 失败源于 macOS 打包产生的 `._*.tif` 元数据文件使文件数误判为 72/72；未开始训练。保留失败日志，在同一 Pod 内建立过滤元数据的干净副本后，以**相同科学配置**重试。正式 GPU preflight 看到唯一 A40，TensorFlow GPU 矩阵运算成功，36 对原始 1200×1200 TIFF 校验通过。五折 checkpoint、history、split、阈值和日志分别打包并在 Mac 通过 SHA256 核对；原图、人工 mask、Cetus 作业均未修改。Pod 已删除，`list_pods` 为空，运行 GPU 支出速率为 US$0/hour。2026-10-02 12:01 UTC 查询到的暂列费用为 US$0.810319（GPU US$0.804821、disk US$0.005498）；账单尚未最终结算，不能把此数当成最终费用。

**Reject：**此固定亮背景 hard-negative 采样没有通过任何折的 validation gate，不替换冻结 baseline，也不继续扫描其比例或阈值。研究者在设计前已检查同一批 36 张图的 baseline OOF 错误，故本实验仍属探索性；初始分层 fold map 使用过所有 36 张 mask 的前景比例，且原 organoid 身份不可恢复，潜在图像相关性不能排除。未来外部未见图像未参与训练/调参。当前无法判定部分亮“负样本”究竟是其他结构、成像伪影还是漏标 Mito。

**下一项最有信息量的实验：**先由熟悉标注规范的人员对预先按前景比例和误报类型分层抽取的亮背景区域做盲法复核，记录独立判断的一致率；不要因为某张 OOF 差图定向修改 mask。该非 GPU 诊断可确认 hard-negative 假设的标签前提。若亮区域确为背景，再预先固定一个训练折内的温和局部亮度/对比度增强作为**单因素**五折候选，继续用原 gate 和 outer-test 隔离验证；在标签语义不清前不应继续加大负样本权重。
