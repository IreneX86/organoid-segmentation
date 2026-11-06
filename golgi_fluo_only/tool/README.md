# 只用 Fluo 预测 Golgi（实验性工具）

输入一个 Z-stack 的 Fluo 图像，输出每层的 Golgi 概率图和 mask。不需要 Golgi 通道。

## 这是什么，可信到什么程度
- 6 个 U-Net 取平均：
  - 3 个“2.5D”模型，输入为 z−1、z、z+1 三层 Fluo；
  - 3 个“C7”模型，输入为 z−3 到 z+3 七层 Fluo。
- 阈值固定为 0.45。
- 训练标签来自“看 Golgi 通道的 U-Net”（教师模型）生成的伪标签，**不是人工真值**。
- 集成方案和阈值只用留出细胞球上的伪标签来挑选，14 张人工标注只在最后检验了一次：

| | 整图 Dice | 近 Golgi 区域 AUC |
|---|---:|---:|
| 本工具（只看 Fluo） | 0.218 | 0.885 |
| 教师模型（看 Golgi 通道） | 约 0.78 | — |

- **结论：它能大致找到 Golgi 在哪一带，但形状不准。不要用它测 Golgi 的面积、形态或数量。**
- 训练数据里 Golgi 通道只在部分细胞里亮，模型也会在其他细胞里画出“像 Golgi 的区域”。这些预测无法用现有数据核对。
- 只在 1200×1200 的 MCF-7 细胞球 Fluo 图像上训练和检验过，同一套显微镜设置。其他细胞类型或拍摄条件需要重新验证。

## 怎么用
需要 Python 3 和 `numpy`、`tifffile`。如果装了 TensorFlow，会自动用它，速度快很多；没装就用纯 NumPy，每层约 10 分钟（6 个模型）。

```bash
# 输入可以是一个文件夹（单层 TIFF，按文件名排序就是 Z 顺序），也可以是一个多页 TIFF
python golgi_predict.py /path/to/cell_xxx/fluo --out /path/to/output_folder
# 只算部分层（从 1 开始编号）
python golgi_predict.py /path/to/cell_xxx/fluo --out out_folder --planes 30,40-45
```

- 输出文件夹必须是空的或不存在，工具不会覆盖已有文件。
- 首尾层缺少上下相邻层时，用最近的一层补。
- `golgi_prob_*.tif`：概率 × 255（uint8）。
- `golgi_mask_*.tif`：0/255 的 mask。
- `manifest.json`：参数、模型和输入文件的 SHA256，方便日后复查。
- 启动时会校验 `models/` 里 6 个模型文件的 SHA256，文件损坏会直接报错。

## 文件
- `golgi_predict.py`：工具本体。
- `models/*.npz`：6 个模型的权重（16 位浮点，每个 13 MB）。已包含在仓库中（共约 86 MB），工具启动时会按 `MODELS.json` 校验 SHA256。
- `models/MODELS.json`：每个模型的来源和 SHA256。
- 训练代码见本仓库 `golgi_fluo_only/training/`；完整结果（模型、概率图）只保存在本地 `outputs/golgi_fluo_student_v*_2026110*/`。
