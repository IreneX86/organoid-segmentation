# 脂滴自动分割结果（供人工复核）

此目录保存 **10 个细胞、1,330 张**脂滴（LD）自动预测 mask 的压缩交付件。每个 `cell_*/fluo_label.zip` 解压为 `fluo_label/`，其中的 TIFF 与该细胞 `fluo/` 输入同名、像素坐标一一对应，均为 1200×1200、`uint8`、背景 0／前景 255。每包另含脱敏的 `PORTABLE_MANIFEST.json`、`SHA256SUMS.txt` 和说明文件；相邻的 `fluo_label.zip.sha256` 校验整个 ZIP。压缩包**不含原始荧光图像、人工标签、模型权重、运行日志或本机绝对路径**。

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

这十组当前的 `fluo_label` 使用相同的**五折 focal U-Net 投票**：逐图 p1–p99.8 归一化，512×512 重叠滑窗（重叠 128 像素），各折采用锁定的验证阈值 `[0.65, 0.80, 0.60, 0.35, 0.50]`，至少 3/5 折为前景才输出 LD。首组的旧预测已另存本地 `fluo_label_previous`，本目录**没有**归档旧预测。其余九组共 1,230 张在一次 Secure Cloud A40 批量任务中完成；本地逐张验证文件名、尺寸、像素值与 SHA256 均通过。每组的模型哈希和 mask 哈希见 ZIP 内的脱敏清单。

**科学边界：**这是项目负责人目视偏好的人工复核版本，不是人工描绘的 ground truth，也不是通过独立外部数据验证的新生产模型。此前预注册五折验证门限均未选择 focal 候选；投票集成本身没有独立的 Dice／IoU 性能估计。不能把这些 mask 混入 `label_ref` 的人工训练标签，或用这些被审阅的图像声称无偏外部准确率。[实验报告](../../reports/ld_focal_gamma2_20261014/README.md)和[首组复核记录](../../reports/ld_focal_human_preferred_inference_20261015/README.md)保留该区分。

在每个细胞目录运行 `shasum -a 256 -c fluo_label.zip.sha256` 检查下载包；解压后可按 `SHA256SUMS.txt` 再核对逐张 mask。归档由 [`code/package_ld_auto_labels.py`](../../code/package_ld_auto_labels.py) 从本地已验证的当前输出生成，不会修改原始 TIFF 或输出 mask。
