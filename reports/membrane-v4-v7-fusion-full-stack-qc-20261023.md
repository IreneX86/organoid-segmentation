# Membrane v4 + v7 保守融合：全栈复核

日期：2026-10-23。此结果仅用于人工审查，不是独立验证或生产模型升级。原始 `Memb.tif`、`fluo.tif`、人工标签、v4/v7 checkpoint 和既有预测保持不变。

## `cell_21941_NZ_330`：已完成的 330 层结果

复用同条件 local11 的 v4 和 v7 完整预测，不重新训练或推理。输出为 330×1200×1200、`uint16` 的三维实例 TIFF；SHA256 为 `35349b39b78542e624f609c63919d8ed8a38efe8e9fd3f43345357dff03e0592`。v4 原有的 169,723,460 个前景体素及其 ID 均逐像素保留；新增 1,908,995 个原本属于 v4 背景的体素。v4、v7、融合结果分别有 210、189、211 个三维 ID。这个数量与人工观察约 212 个暗核接近，但不是逐核对应验证。

逐段复核了 19 个接纳的 v7 连续段（155 个逐层事件）：每段新增区域在三维上有一个持续连通块，18 段接入唯一 v4 轨迹，1 段生成新 ID 1038；原 v4 像素/ID 在这些层均未改变。一段 z166–169 同时指向 v4 ID 138 与 141，已保持原 v4、不予补入。因而程序化检查未发现补入造成的**v4 ID 碰撞或双锚连接**；这不等于已经证明每个补入实例在生物学上都是单个细胞。

`fluo` 暗区只作候选线索：16 段仅有一个跨层持续暗区候选，3 段出现多个候选（z168–170 的 v7 #130、z190–216 的 v7 #118、z215–217 的 v7 #127）。这些多暗区可能来自真实多核/粘连，也可能来自纹理或阈值造成的碎片；没有逐核人工实例标签，不能据此自动拆分或宣称错误合并。详细结果在 [逐段 QC JSON](../runpod_results/membrane_v4_base_v7_additions_20261023/qc_19_runs.json) 和 [CSV](../runpod_results/membrane_v4_base_v7_additions_20261023/qc_19_runs.csv)。

![v4（青）、v7（绿）、融合（黄）以及原始 Memb/fluo 的固定 Z 层对照](../runpod_results/membrane_v4_base_v7_additions_20261023/fusion_3d_qc_comparison.png)

可在 Fiji 中打开 [完整三维实例 TIFF](../runpod_results/membrane_v4_base_v7_additions_20261023/memb_instances_v4base_v7add_safe.tif) 作测量；[8 层 RGB 叠加 TIFF](../runpod_results/membrane_v4_base_v7_additions_20261023/fusion_overlay_selected_z_rgb.tif)仅供显示，页码对应原始 Z=133、168、166、200、216、232、265、300，见 [页码表](../runpod_results/membrane_v4_base_v7_additions_20261023/fusion_overlay_selected_z.json)。RGB 叠加图上的黄色填充表示 v7 新补入的区域，不保留实例数值，不能拿来做测量。可用 [napari 查看器](../experiments/view_membrane_v4_v7_fusion_20261023.py)同时按需加载原始双通道、v4、v7、融合和“新增像素”图层；它按页延迟加载，不更改源文件。

现有人工修正区域的探索性配对结果见[上一轮报告](membrane-v4-base-v7-conservative-additions-20261023.md)。这些区域可能参与过历史模型开发且标注不完整，不能用于宣称独立泛化。尤其“Z 身份一致率”只评估已有人工对应关系，不会自动发现连续但错误合并的细胞。当前融合保留为 **review candidate**，三个多暗区段列入人工重点复核。

更严格的 [Z-presence 审计](../runpod_results/membrane_v4_base_v7_additions_20261023/track_continuity_qc.json)显示：v4 的 210 个 ID 和融合的 211 个 ID 均没有缺层后以同 ID 重现的现象；补入没有给原 v4 ID 新增 Z 缺口。此检查能发现号码断续，却不能发现持续存在的错误合并。

## 规则开发外应用：`CF/cell_21422_NZ_100`

该完整 100 层类器官此前没有同条件的 v4/v7 输出，因此在单张 Runpod Secure A40 上用**原始、哈希锁定**的 v4/v7 checkpoint 各运行一次 2D 分批预测和全局 stitching。两者都使用相同的双通道顺序 `Memb, fluo`、Cellpose 4.2.1.1、global-Z local11 归一化和原推理参数。GPU 预检通过；结果下载后按压缩包、模型、源 TIFF、输出 TIFF SHA256 和几何尺寸逐项校验通过。仅运行推理，**没有训练**。运行归档和[本地校验记录](../runpod_results/membrane_v4_v7_cf100_20261023/local_verification.json)在忽略 Git 的本地目录，原始文件和旧 v5 预测未改。

将完全相同的候选筛选与受约束 3D ID 分配用于这套新预测：138 个低重叠逐层候选中，只有 **2 段／6 个逐层实例**满足预定时间及信号门槛，分别为 z65–67 的 v7 #93 接入 v4 #157，以及 z97–99 的 v7 #173 接入 v4 #235；其余均不自动补入。新增 **47,478 个** v4 背景体素。v4、v7、融合各有 **245、237、245** 个全栈 3D ID：这次没有新建完整细胞 ID，只延长了两条 v4 已有轨迹。两段的新增部分各是一个三维持续连通块，暗 `fluo` 内部各有一个持续候选区；这是支持线索，**并非人工核确认**。没有双锚冲突；v4 原有 ID 完整保留，两份结果均无新增 Z 缺口，见[融合配置](../runpod_results/membrane_v4_v7_cf100_20261023/fusion/safe_config.json)、[逐段 QC](../runpod_results/membrane_v4_v7_cf100_20261023/fusion/qc_runs.json)和[Z-presence QC](../runpod_results/membrane_v4_v7_cf100_20261023/fusion/track_continuity_qc.json)。

![CF 100 层中两处补入：Memb、fluo、v4、v7 和融合](../runpod_results/membrane_v4_v7_cf100_20261023/fusion_cf100_comparison.png)

[CF 完整 3D 实例 TIFF](../runpod_results/membrane_v4_v7_cf100_20261023/fusion/memb_instances_v4base_v7add_safe.tif) 为 100×1200×1200、`uint16`，SHA256 `5eccf70b322eb8ca2dc751e78556312442fa56853132d9be8a90e55952f982c4`。可用 Fiji 打开[选定 Z 层 RGB 叠加 TIFF](../runpod_results/membrane_v4_v7_cf100_20261023/fusion_cf100_selected_z_overlay_rgb.tif)，其真实 Z 序号见[页码表](../runpod_results/membrane_v4_v7_cf100_20261023/fusion_cf100_overlay_pages.json)。完整 v4、v7 和融合结果可同时用上述 napari 查看器打开，传入 `--cell`、`--v4`、`--v7`、`--fusion` 路径。RGB TIFF 只作显示；定量测量应使用原始 3D 实例 TIFF。

此 CF 样本没有在本轮提供完整、独立的逐细胞人工真值，因此只能做**实际应用和视觉诊断**，不能给出它的独立 Recall、Merge/Split 或核–细胞一对一准确率。虽然规则开发只用先前 MCF-7 的融合案例，v4/v7 checkpoint 的更早训练/选模来源仍需另行确认；这里也不称 CF 为独立模型测试集。此实例表明当前规则在另一类器官上相当保守，主要填补既有轨迹的缺层，而非大量找回新细胞。

Runpod Pod 仅用于这一次 CF 推理，结果下载并校验后已 **TERMINATE**，随后 `list_pods` 为空。创建时报价 US$0.59/小时；最终逐 Pod 账单在检查时尚未入账，因此本报告不把时间乘单价的估算冒充实际费用。
