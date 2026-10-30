# ER 线间背景单因素实验 — 2026-10-29

## 结论
不升级为生产模型。增加线间背景候选后，大片阳性有所减少，但仍未形成可信的 ER 细线网络；收益不稳定，并伴随召回下降。保留全部结果，不自动继续训练。

## 固定实验
仅对32张训练裁剪增加75,178个 provisional negative 像素；原有阳性、输入 Fluo、8张验证裁剪及其标签保持不变。模型、loss、阈值0.5和训练配置固定。验证来自同一个历史上已查看的样本，属于探索性评估，不是独立测试。

单台 Secure Cloud A40，训练19 epochs，最佳11，EarlyStopping结束。训练及评估约30.64秒，不含环境安装。原始 checkpoint 保留；新旧模型都采用相同的仅训练输入 BN 统计校准，没有梯度更新或验证集校准。

## 配对结果
以下为验证已知标签像素汇总的 pseudo-label agreement，不是人工真值准确率，也不是逐图均值。

|指标|原模型校准后|新模型校准后|
|---|---:|---:|
|Dice|0.8170|0.8198|
|IoU|0.6906|0.6946|
|Precision|0.7272|0.7495|
|Recall|0.9322|0.9046|
|全部像素预测阳性比例|75.04%|70.03%|

8张裁剪仅4张 Dice 上升。已知阴性上的 FP 从43,560降至37,645，FN从8,441增至11,887。不能将未标注像素的阳性一律算作假阳性。视觉上仍有大片实心预测，细网结构不足，不能把微小 Dice 上升解释为目标达成。

## 文件和校验
- 完整训练结果：`../outputs/er_interline_student_run_20261029/extracted/results/student/`
- 下载归档和 SHA256 已核验，checkpoint ZIP及40张概率数组检查通过，checkpoint可加载。
- checkpoint SHA256：`d3a3ffd3d3509debf77ef658ba20ced0c5af67fda652a9efceb66dbe9f7dade9`
- [八张配对图](../outputs/er_interline_bn_calibrated_20261029/paired_all8.png)
- [小型结果](../outputs/er_interline_bn_calibrated_20261029/paired_comparison.json)
- 协议：`../docs/er_interline_ablation_protocol_20261029.md`

## 费用和清理
Pod `[pod-id-redacted]` 已停止并终止。Runpod Billing 显示GPU/vCPU/Storage/Endpoint均为0，实时支出US$0.00/hour，余额US$4.79。账户显示余额从US$4.86变为US$4.79，本次约US$0.07（显示余额差额，非逐项账单）。

## 下一步
不继续扫描阴性比例。先做单个训练裁剪的拟合能力诊断，区分优化/归一化设置与监督信息不足；只有能在训练裁剪重现细线结构后，再考虑跨样本泛化。该诊断尚未启动，不能由当前结果断言 Fluo 一定不能预测 ER。
