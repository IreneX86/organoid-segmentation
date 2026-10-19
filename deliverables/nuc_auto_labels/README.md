# Nuc 自动分割结果交付包

本目录保存 10 个细胞的 **1,330 张细胞核自动预测 mask**。每个细胞有独立的 `Nuc_label.zip`；解压后得到 `Nuc_label/`，其中 `fluo_*.tif` 与该细胞的原始 `fluo/` 输入一一同名、同为 1200×1200 像素，坐标对应。mask 为单通道 `uint8`、`MINISBLACK`，背景 0（黑），细胞核 255（白）。

| 细胞 | mask 数量 |
| --- | ---: |
| `cell_22375_NZ_100` | 100 |
| `cell_22376_NZ_100` | 100 |
| `cell_22377_NZ_120` | 120 |
| `cell_22378_NZ_150` | 150 |
| `cell_22379_NZ_150` | 150 |
| `cell_22380_150` | 150 |
| `cell_22381_140` | 140 |
| `cell_22382_140` | 140 |
| `cell_22383_140` | 140 |
| `cell_22385_140` | 140 |
| **合计** | **1,330** |

这些 mask 由冻结的 2D Nuc U-Net checkpoint 加上负责人选定的人工复核后处理方式生成：逐图 p1–p99.8 归一化、512×512 滑窗（重叠 128）、概率阈值 0.03、强核参考阈值 0.45、去除小于 500 像素的区域、填洞、受约束的半径 4 闭合、边缘平滑，以及仅对明确的双核粘连作平滑分离。具体方法见[复核处理报告](../../reports/nuc_review_postprocessing_20261019.md)和[预测代码](../../code/nuc_predict_review.py)。本次使用单张 Runpod Secure Cloud A40 GPU 批量推理，并逐细胞校验、下载和发布。

**这些是未经逐张人工确认的模型预测，不是人工 ground truth。** 目标图像没有独立的人工 mask，因此不能从本交付包推断外部 Dice 或临床／生物学准确率。低阈值有助于人工复核漏检核，也可能产生假阳性；请保留人工审核步骤，不要将这些 mask 直接混入 `label_ref` 训练标签。

每个 ZIP 仅包含生成的 mask、脱敏的 `PORTABLE_MANIFEST.json`、说明和逐张 `SHA256SUMS.txt`；不包含原始 `fluo` 图像、人工标签、checkpoint、本机绝对路径、完整运行日志或凭据。各 ZIP 旁的 `.zip.sha256` 和本目录的 `SHA256SUMS.txt` 校验整个压缩包，`PACKAGE_INDEX.json` 记录每包的数量、大小及 SHA256。解压到新目录，避免覆盖已有结果。

在本目录运行 `env LC_ALL=C shasum -a 256 -c SHA256SUMS.txt` 可核对全部 ZIP；解压后还可使用各包内的 `Nuc_label/SHA256SUMS.txt` 核对逐张 mask。
