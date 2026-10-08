# Mito 自动分割结果

此目录按项目负责人明确要求，在**私有** GitHub 仓库保存模型生成的 Mito 二值 mask 压缩包。原始显微 TIFF、人工标签、模型 checkpoint 和运行日志不在这些压缩包内，也不提交到 Git。

每个 `cell_*/Mito_label.zip` 解压后得到 `Mito_label/`：与该细胞 `Mito/` 输入同名、同为 1200×1200 像素网格的 TIFF；uint8 背景为 0、前景为 255。附带 `prediction_manifest.json`、`AUTO_LABELS_README.txt` 和逐文件 `SHA256SUMS.txt`。每组压缩包旁的 `Mito_label.zip.sha256` 用于校验整个归档。

目前已归档 `cell_22375_NZ_100`（100 张）、`cell_22376_NZ_100`（100 张）、`cell_22377_NZ_120`（120 张）、`cell_22378_NZ_150`（150 张）、`cell_22379_NZ_150`（150 张）、`cell_22380_150`（150 张）、`cell_22381_140`（140 张）、`cell_22382_140`（140 张）、`cell_22383_140`（140 张）和 `cell_22385_140`（140 张），共 **1,330 张**自动预测 mask。`organoid_data` 中不存在 `cell_22384`。

这些是**未经过人工确认的自动预测**，不得直接视为 ground truth 或混入 `label_ref` 的 36 对人工标注数据。新图没有人工 mask，因此目前无法测量其真实 Dice。所有组均使用冻结的 Mito 2D U-Net fold-0 checkpoint（SHA256 `66db8edc3006a9b70fdd75901f6f5e4a40359699f7a372d28f054d8dec58ecd3`）、验证集阈值 0.45、逐图 1–99.8 百分位归一化，以及 512 像素 patch / 128 像素重叠的滑窗概率融合。该 checkpoint 是内部交叉验证模型，并非用全部 36 张人工标注图训练的 production model。

在 Mac 上可用 `shasum -a 256 -c cell_*/Mito_label.zip.sha256` 核对下载，然后用 `unzip` 解压。原始数据仍留在本地 Desktop 项目目录；Git 中只包含预测 mask。
