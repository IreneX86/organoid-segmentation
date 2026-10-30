# ER 双通道探索性分割

当前试验使用的是**传统图像处理规则，不是 U-Net，也没有训练过 ER 模型**。输入为同一个细胞目录下配准的 `ER.tif` 和 `fluo.tif` 三维 TIFF；每个 Z 层独立生成二维 ER mask。输出沿用原始 Z/Y/X 顺序，背景为 uint8 `0`，候选 ER 为 `255`。

## 处理方法

1. 对 Fluo 做两个尺度的高斯平滑之差，提取粗略的细胞结构支持区；闭合、填洞、去小区及扩张只用于限定 ER 候选所在范围。
2. 对 ER 通道计算轻度平滑图与大尺度背景图之差，并使用相对背景对比度和绝对残差阈值得到候选像素。
3. 去掉小于 8 像素的连通片。用 LoG 找亮斑；如果其对应候选连通片面积较大、形状紧实，且 Fluo 中没有相应的局部结构对比，则将其作为疑似圆形杂质排除。
4. 人工已复核的特定图可以另给 ROI JSON，删除该图中被圈定的连通片。这个步骤**不会默认作用于其他层或其他细胞**。

具体数值固定在 [`code/er_classical_segment.py`](../code/er_classical_segment.py) 的 `ERParameters` 中。它们来自一张图的探索性试验，尚未在所有类器官上验证。真正 ER 也可能呈点状；批量结果须抽查弱信号、圆形 ER、暗背景和 Z 方向连续性，不应直接当作人工真值。

## 运行

使用具有 `numpy`、`scipy`、`scikit-image` 和 `tifffile` 的 Python 环境。输入根目录既可以是 ER 数据总目录，也可以是单个细胞目录。输出目录必须尚不存在；程序会镜像细胞子目录，逐页写入 `ER_mask.tif` 和带参数、Z 索引、SHA256 的 `ER_mask_manifest.json`，不会改动输入文件。

先测试两个 Z 层（索引从 0 开始）：

```bash
python code/er_classical_segment.py \
  --input-root "$HOME/Desktop/Organoid/ER/MCF-7_20260814/cell_20260814_23298_NZ_50" \
  --output-root "$HOME/Desktop/Organoid/ER_preview_test" \
  --z-start 36 --z-stop 38
```

之后在**新的**输出目录运行所有细胞和完整 Z-stack：

```bash
python code/er_classical_segment.py \
  --input-root "$HOME/Desktop/Organoid/ER" \
  --output-root "$HOME/Desktop/Organoid/ER_masks_classical_v1"
```

如需**仅对已经人工圈定的试验图**复现其额外去噪，输入总目录并附加：

```bash
--review-rois experiments/er_pilot_z037_review_rois.json
```

这份 JSON 只指定 `MCF-7_20260814/cell_20260814_23298_NZ_50` 的 zero-based `z=36`。默认不加载它。若传入单个细胞目录，JSON 中对应键须为 `"."`；这样可以防止不同输入范围下误用相同坐标。

## 已验证的范围

在一张 1200×1200 的已复核图上，自动规则删除了 5 个连通片，用户圈选的特定区域又删除了 7 个。代码使用该 ROI 时，与已认可的 `ER_mask_z037_denoised_v2.tif` **像素完全一致**（差异像素 0）。这是实现回归检查，并非模型准确率或独立生物学验证。
