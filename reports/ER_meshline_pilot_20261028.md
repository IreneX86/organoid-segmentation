# ER 网状细线候选对照（2026-10-28）

用户明确要求标出亮区内部可辨认的 ER 网状线条，而不只是圈出外缘。本次保留现有输出，另以 `experiments/er_meshline_pilot.py` 对八张代表切片生成独立候选。输入为原始 ER、Fluo 和旧 mask；旧 mask 仅提供宽松空间上下文，新增线段仍必须通过 ER 通道的多尺度亮线响应及局部对比度条件。输出中心线约 1–3 像素宽，以洋红色叠加在原始 ER 上供审查，纯 mask 为 0/255 的 uint8 TIFF。

结果在 `outputs/er_meshline_pilot_20261028/`。每个类器官目录包含 `comparison_zXXX.png` 和 `ER_meshline_candidate_zXXX.tif`，`manifest.json` 记录输入组、参数审计及输出 SHA256。八张 1200×1200 候选 TIFF 均通过形状、二值、SHA256 和对照图存在性校验。

代表对照：

- [明亮球体 23298 Z39](../outputs/er_meshline_pilot_20261028/cell_20260814_23298_NZ_50/comparison_z039.png)
- [较暗类器官 23966 Z55](../outputs/er_meshline_pilot_20261028/cell_20260904_23966_NZ_100/comparison_z055.png)
- [另一成像条件 23961 Z44](../outputs/er_meshline_pilot_20261028/cell_20260904_23961_NZ_100/comparison_z044.png)

视觉上，此候选比旧 mask 更细，在部分亮区内部形成细线网状结构，也补入旧 mask 之外的可见亮纹。但部分新增线条可能只是背景纹理，部分模糊区域和弱 ER 仍不确定。像素数变化只描述几何差异，不能作为 Dice、召回率或生物学真实性证据。未生成或宣称人工真值，也未替换现有 mask 或修改原始 TIFF。适合作为下一步逐图人工审查的候选；在没有可信 ER 人工校正区域前，不应批量推广为正式标签。
