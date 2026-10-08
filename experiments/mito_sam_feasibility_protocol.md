# Mito SAM 本地可行性方案 — 2026-10-08

本轮仅作 CPU、无需训练的探索性可行性测试，不选择/替换 frozen baseline，不启动 Runpod。当前 baseline OOF 已被反复查看，以下四图结果不是新的独立测试集，也不是五折 SAM 成绩。

## 在运行预测前固定

- 从冻结 OOF CSV 按前景 ≤5% / >5% 分组，再在组内以 baseline Dice 中位数划分较低/较高 Dice；各格按 SHA256(`20261008-sam|filename`) 升序取一张，共四张。抽样使用既有图像级结果只用于覆盖不同错误情形，不使用 SAM 结果选图。
- 两个预训练模型：官方 micro-SAM `vit_b_lm` 当前注册权重，通过官方 SAM predictor 执行（**不冒充完整 micro-SAM AIS/APG**）；官方 SAM2.1 Hiera Tiny 图像 predictor。MedSAM 和 EM 专用方法先进行适用性与可复现性审查，不额外扩大模型搜索。
- 每个模型比较同样的两种全自动 box 来源：①输入自身逐图 1–99.8% 归一化、Gaussian σ=1 后 Otsu 二值分割；②冻结的对应 outer-fold baseline OOF 二值预测。第二种是已训练 U-Net+SAM refinement，不能称作纯 zero-shot pipeline。
- 连通域按 8 连通、面积至少 10 原像素，bbox 每边扩展 4 px 并裁在 patch 内；每个连通域只提供一个 box，无人工点击、无 GT prompt、无真实实例假设、不筛选 predicted IoU、不事后调阈值。Otsu proposal 二值图本身作为无 SAM 对照保留。
- 原图 1200×1200 只做 512×512 patch，stride=384，补最后一个起点；各 patch 内 SAM 的标准 1024 encoder 缩放属于模型预处理。整张图不缩到512；输出恢复每个 patch 原坐标，在重叠区平均概率，固定0.50二值化，最终输出同名同尺寸 uint8 0/1 TIFF。
- 每个 patch 的多个 prompt mask 取 sigmoid(logit) 最大值，再拼接；无 prompts 的 patch 为零。micro-SAM 使用 single-mask；SAM2 使用官方 single-mask 默认配置。所有模型 eval、inference_mode、CPU float32、最多4线程，批量8 prompts。
- 推理命令没有 mask/label 路径参数；保存全部预测、输入/模型/prompt SHA 和完成标记后，独立评分命令才接收 GT 目录。不能因结果不好换 prompt 规则/阈值/模型重测。技术错误可以修复并记录。

## 评分与解释

同四图配对比较 Dice、IoU、precision、recall、FP/FN 像素、距GT>10px FP、2px tolerance boundary F1、逐图/低前景表现；报告时间、CPU/RAM峰值、模型下载大小及本次云支出US$0。小样本只作描述，不能给可靠泛化提升结论。

人工 prompts 与自动 prompts 分开：本轮没有人类复核者提供 input-only prompts，因此不报告人工 prompt 实测分数，不用 GT 生成“人工”boxes，也不把模型的内部 predicted IoU 当真实 IoU。后续若人工交互测试，先独立只看原图记录 clicks/boxes及耗时，锁定后评分；应报告按人计时和完整图覆盖率。

只有多张图显示明确、同方向的质量优势且无低前景/recall严重代价时，才考虑预注册五折微调。小幅局部改善不能升级 baseline。若未满足则不设计/运行大规模训练；保留失败证据和可复现脚本。
