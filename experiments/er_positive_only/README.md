# 只用 ER 阳性细胞训练 Fluo-only U-Net：数据与方案（2026-10-31）

## 数据（dataset_v2/）
包含 6 个样本：MCF-7 的 23298、23300，WT 的 23961、23963、23965、23966。每个样本在 Z 中段 10–90% 范围内每 5 层取 1 层，共 86 层。23299 和 23962 的 TIFF 单个文件超过 400 MB，无法上传到云端，所以暂未纳入。

较大的样本拆成了多个 `<样本>_partN.npz` 文件，每个文件最多 6 层，训练脚本会自动合并。每个 npz 文件包含：
- `fluo`：模型输入，也是唯一的输入。
- `mask`：ER_mask_v2 伪标签，由 ER.tif 生成，不是人工真值。
- `region`：ER 阳性细胞区。只在区内计算 loss 和打分；区外 ER 状态未知，一律忽略。
- `z`：切片编号，从 0 开始。

同名的 json 文件记录了源 TIFF 的 SHA256 和生成参数。`qc_overview.png` 是质检图，每个样本取中间一层。

## 基线（h_baseline_v2.json）
梯度提升模型，只用 Fluo 特征，留一样本交叉验证，只在阳性区内打分：平均 AUC 0.704，AP 0.415，阳性占比 0.265。

注意：Dice 和 2 px 容差 F1 不能作为判断依据。把阳性区全部判为阳性，Dice 就有约 0.42、容差 F1 约 0.60。

## 训练脚本（code/train_positive_only.py，PyTorch；可在 Mac 的 MPS、NVIDIA GPU 或 CPU 上运行）
示例：`python code/train_positive_only.py --data dataset_v2 --out runs_v2 --steps 3000`

- 固定 3000 次更新，不做早停，不挑选 checkpoint，阈值固定为 0.5。
- 停止条件在运行前已经写定：平均 AP ≥ 0.515、AUC ≥ 0.80，并且在 6 个留出样本中至少 5 个优于基线，才继续这条路线；否则停止，不再调参。
- 该脚本**尚未实际运行验证**：当前云端环境装不了 PyTorch。第一次请先用 `--folds 23963 --steps 200` 做冒烟测试。
