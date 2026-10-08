# SAM 类方法的 Mito 适用性核查

核查日期：2026-10-08。本文是文献/官方代码审查，不把其他数据集的分数当作本项目效果。

## 优先本地测试的两个模型

**micro-SAM。** 官方工具包含交互式 SAM、AMG、额外 decoder 的 AIS，以及从该 decoder 生成 prompts 的 APG。当前 LM generalist 面向光学显微细胞/细胞核；EM organelles 面向电镜线粒体等。荧光 Mito 不等于 EM 线粒体。当前测试采用 LM ViT-B 权重和标准 box predictor，**不是完整 AIS/APG benchmark**。官方注册权重来自 BioImage.IO `diplomatic-bug/1.2/files/vit_b.pt`，xxh128 `fe9252a29f3f4ea53c15a06de471e186`。参考：[官方模型与模式说明](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html)、[模型注册源码](https://computational-cell-analytics.github.io/micro-sam/micro_sam/util.html)。

**SAM2.1。** 官方 Hiera image encoder、prompt encoder、mask decoder 支持静态图像；视频 memory 能力对本项目打乱的独立 TIFF 不适用。选择 Tiny 作为资源受限 CPU 探针，只用 image predictor。不可将 Tiny 的结果推广到所有 SAM2 大小/所有 prompts。参考：[官方代码与 checkpoint](https://github.com/facebookresearch/sam2)。

## 文献审查但未本地运行

**MedSAM。** 以 box prompt 适配医疗图像，训练数据以 CT/MRI/内镜等为主。原论文明确讨论分支血管结构和 box 歧义的限制。它提供了领域适配范例，但“medical”并不自动意味着适合荧光细长线粒体；本轮不增加第三个大模型。参考：[MedSAM 原论文](https://www.nature.com/articles/s41467-024-44824-z)、[官方实现](https://github.com/bowang-lab/MedSAM)。

**SAM for Robust Mitochondria Instance Segmentation in Fluorescence Microscopy（2026-05，CVPR workshop）。** 合成荧光数据微调 SAM，推理从输入阈值产生 prompts；报告冻结 ViT-B encoder、只训练 mask decoder 的方案。仅两张人工标注真实图用于实例评估；其实例匹配 Dice/precision/recall 与本项目像素 semantic 指标不同。使用80 nm像素尺度预处理，不能在未知本项目标定时照搬尺度。本文与公开检索未定位可核实的官方可下载专用 checkpoint，因此没有假称复现。参考：[全文](https://arxiv.org/html/2605.31284v1)。

**WeakMitoSAM（2026）。** 面向 EM 的稀疏点监督，CAMP 聚合 SAM proposals 后进行 LoRA 等适配；它依赖实例点信息，不能把 GT 生成的测试点当作自动推理。2026-10-08 GitHub contents API 返回官方仓库根目录仅 `README.md`，无法按现有公开实现复现完整方法。参考：[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13178590/)、[官方仓库](https://github.com/WilliamLee30/WeakMitoSAM)。

**ScSAM（出版页面标注2026-08-25，ECAI 2025卷）。** EM BetaSeg 多类任务，融合冻结 SAM/MAE 特征及可学习 class prompts，说明语义分类和局部纹理的重要性。但依然需要训练；双编码器和EM任务不支持本轮直接迁移。参考：[原论文](https://journals.sagepub.com/doi/10.3233/FAIA251278)。

**Spatial-SAM（CVPR 2026）。** 基于连续3D EM、空间记忆和半监督，使用标注切片模拟交互结果。该前提与独立、无可靠Z顺序的TIFF冲突，排除本轮。参考：[CVPR原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Spatial-SAM_Spatially_Consistent_3D_Electron_Microscopy_Segmentation_with_SDF_Memory_CVPR_2026_paper.pdf)。

## 评价身份必须分清

- 无本项目训练、input-only生成提示：本轮 Otsu→SAM，包含传统proposal机制；不能单独归因于SAM。
- U-Net OOF→SAM：全自动，但继承了本项目监督训练，是两阶段refinement pipeline。
- 人工 prompts：需要仅看输入的独立点击/框及人工耗时；本轮没有人类交互实测，不能报告其Dice。
- 测试GT框/点：带目标位置答案的oracle条件；本轮不运行，更不混入自动结果。
- 完整 micro-SAM AIS/APG、MedSAM、上述专用论文权重：本轮没有实测，不赋分。

对 SAM 的价值判断应限定到真实运行的模型+提示+拼接方案。测试低分不能排除未来领域微调，局部高分也不足以证明新模型在36张图的独立泛化增益。
