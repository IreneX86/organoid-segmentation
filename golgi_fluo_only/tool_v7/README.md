# 只用 Fluo 预测 Golgi（实验性工具，v7，2026-11-09）

输入一个 Z-stack 的 Fluo 图像，输出每层的 Golgi 概率图和 mask。不需要 Golgi 通道。

## 是什么、可信到什么程度
- 3 个 U-Net（按细胞球 3 折训练）取平均，输入为 z−3 到 z+3 七层 Fluo，阈值 0.5。
- v7 训练：学习教师模型（看 Golgi 通道的 U-Net）的**概率图**（知识蒸馏），训练 20000 步、余弦学习率、亮度/对比度/模糊/噪声增强；预测时 8 方向测试时增强（4 个旋转 × 翻转）。
- 训练标签是教师模型的伪标签，**不是人工真值**。
- 留出数据上的检验（36 个留出层 + 14 张人工标注，Golgi 附近 40 px 内的 Dice，阈值 0.5）：

| | 对教师伪标签 | 对人工标注 |
|---|---:|---:|
| 本工具 v7 | 0.422 | 0.412 |
| 旧工具里的 v3 模型 | 0.380 | 0.386 |
| 教师模型（看 Golgi 通道） | — | 约 0.76 |

- **结论：能大致找到 Golgi 的位置，形状只是近似。不要用它精确测量 Golgi 的面积或形态。**
- Golgi 通道只在部分细胞里染上，模型在其他细胞里画出的 Golgi 无法用现有数据核对。
- 只在 1200×1200 的 MCF-7 细胞球 Fluo 图像上训练和检验过（同一显微镜设置）。

## 怎么用
需要 Python 3、`numpy`、`tifffile`。装了 TensorFlow 会自动使用（快）；没装就用纯 NumPy（每层约 3 个模型 × 1–2 分钟；默认不做测试时增强）。

```bash
python golgi_predict.py /path/to/cell_xxx/fluo --out /path/to/output_folder
python golgi_predict.py /path/to/cell_xxx/fluo --out out --planes 30,40-45      # 只算部分层（从 1 开始）
python golgi_predict.py /path/to/cell_xxx/fluo --out out --tta 8                # NumPy 也做 8 方向增强（慢 8 倍，和检验时一致）
```
- 输出文件夹必须为空或不存在。首尾层缺少相邻层时用最近一层补。
- `golgi_prob_*.tif`：概率 × 255；`golgi_mask_*.tif`：0/255；`manifest.json`：参数、模型和输入的 SHA256。
- 启动时校验 `models/` 里 3 个模型文件的 SHA256。

## 复核
- NumPy 版本在一个留出层上与 GPU 原始输出对比：相关 0.9996，mask 一致度（Dice）0.990（差别来自 16 位权重）。
