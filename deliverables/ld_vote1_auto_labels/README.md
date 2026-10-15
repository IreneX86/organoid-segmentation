# LD 1/5 自动预测 mask 交付包

本目录是 **10 个细胞、1,330 张**脂滴自动预测 mask 的独立 1/5 投票交付件。每个细胞的 `fluo_label_vote1.zip` 解压为 `fluo_label_vote1/`，TIFF 与该细胞 `fluo/` 输入同名、尺寸与像素坐标一致，均为单通道 `uint8`、`MINISBLACK`、背景 0、前景 255。每包内的 `PORTABLE_MANIFEST.json` 记录输入和 mask SHA256、模型哈希与阈值；`SHA256SUMS.txt` 可核对解压后的 mask。相邻 `.sha256` 文件及本目录 `SHA256SUMS.txt` 可核对完整 ZIP；`PACKAGE_INDEX.json` 记录十包的数量、大小和哈希。

| 细胞 | mask 数量 | ZIP 字节数 |
| --- | ---: | ---: |
| `cell_22375_NZ_100` | 100 | 365,577 |
| `cell_22376_NZ_100` | 100 | 324,996 |
| `cell_22377_NZ_120` | 120 | 428,127 |
| `cell_22378_NZ_150` | 150 | 491,903 |
| `cell_22379_NZ_150` | 150 | 586,397 |
| `cell_22380_150` | 150 | 471,328 |
| `cell_22381_140` | 140 | 480,611 |
| `cell_22382_140` | 140 | 452,975 |
| `cell_22383_140` | 140 | 374,016 |
| `cell_22385_140` | 140 | 491,792 |
| **合计** | **1330** | **4,467,722** |

五个 focal U-Net 折模型使用锁定的验证阈值 `[0.65, 0.80, 0.60, 0.35, 0.50]`。逐图 p1–p99.8 归一化，512×512 滑窗、重叠 128 像素。**至少一折**投票为前景时输出 LD。这是负责人目视偏好的探索性人工复核版本，**不是人工 ground truth**，投票规则没有独立外部准确率估计。它独立于此前的 3/5 交付包，解压时也使用独立的 `fluo_label_vote1/` 文件夹。

ZIP 只含生成的 mask 和脱敏元数据；不含原始荧光图像、人工标签、模型权重、完整本地预测 manifest、日志、凭据或本机绝对路径。请勿将其当作 `label_ref` 的人工训练标签。由[独立打包脚本](../../code/package_ld_vote1_auto_labels.py)从本地已核验输出生成。

下载后在本目录运行 `env LC_ALL=C shasum -a 256 -c SHA256SUMS.txt` 验证十个 ZIP。若只验证单包，请进入该细胞的子目录，再运行 `env LC_ALL=C shasum -a 256 -c fluo_label_vote1.zip.sha256`。把 ZIP 解压到新的空目录，避免覆盖已有预测文件；ZIP 内的 `fluo_label_vote1/SHA256SUMS.txt` 可继续核对逐张 mask。[旧版 3/5 结果](../ld_auto_labels/README.md)和[生成与验证报告](../../reports/ld_vote1_auto_labels_20261015/README.md)保留完整方法边界。
