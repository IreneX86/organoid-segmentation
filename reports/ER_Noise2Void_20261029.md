# Fluo-only Noise2Void 去噪与 ER 候选分割实验

日期：2026-10-29。结论：**实验完成，但不推荐将本轮 N2V + 简单阈值结果作为正式 ER 标签或训练 Ground Truth。** 原始 TIFF、历史 mask、旧模型全部保留。

## 实验范围和数据检查

检查了现有8份 Fluo 堆栈，共900层，均为 uint16、1200×1200。TIFF 记录为 QYX、没有可信物理 Z-step；依照项目已有连续切片定义处理 Z，但不假设各向同性、不进行3D训练。完整预测对象是 `cell_20260904_23966_NZ_100/fluo.tif`，不是对全部900层批量预测。

该目标100层的代表切片 Z1/25/50/75/100均保存原图、去噪图、mask和overlay。原图采用不变的分位显示范围与NLM/N2V比较，避免单独自动对比度制造改善假象。

八份图像均检查源SHA256。相邻Z相关性较高，样本级中位数约0.981–0.995。高通残差的低梯度区域邻像素相关性仍很高（横向中位数约0.78–0.82）。这提示局部纹理、模糊、重采样或相关噪声可能存在，**无法仅凭这些图像区分其来源，也无法断定独立Poisson或Gaussian噪声模型**。波形噪声估计和对比噪声比仅为代理量，不是真实SNR。没有干净重复采集，因此不报告真实PSNR/SSIM。

## 训练与隔离

采用官方 [CAREamics](https://github.com/CAREamics/careamics) 0.3.5 的原始2D Noise2Void实现，未从零实现N2V。参考[Noise2Void论文](https://arxiv.org/abs/1811.10980)。

- train：23298、23299、23961、23962、23965；validation：23963；本轮目标：23966。
- 旧测试样本23300不参与训练；23966在既往工作中曾被人工查看，不声称此次是盲测。
- 每份训练/验证样本取8张均匀分布切片，每层6个非重叠128×128区域；总计240个训练区域、48个验证区域。训练内部再采64×64patch，不跨样本划分。
- 跨划分的完全重复crop哈希检测通过；这不能证明不同样本绝无同源关系。相邻切片从未跨train/validation。
- 无ER通道、无ER伪mask、无人工mask参与去噪训练。损失只看被遮挡的Fluo像素。
- U-Net depth2、初始16通道；batch16；Adam 0.0004；30 steps/epoch上限，30epochs上限；patience7；seed20261029；CAREamics训练均值/标准差归一化。
- 训练完整运行19epochs，约110秒；最佳为第12epoch，validation blind-spot MSE=0.0111034。验证仅限固定的最多6batches，未对整份验证堆栈做穷尽评估。
- 使用最佳checkpoint进行输出。日志中的SIPSNR以有噪声输入为参考，**不是干净图像PSNR或ER分割准确率**。
- 独立 `.venv-n2v` 安装依赖，未修改旧TensorFlow环境；本轮本地CPU完成，无Runpod费用。

## 传统基线和固定分割

NLM：patch5、search6、h=0.8×wavelet sigma，fast模式，原强度单位。

原图、NLM和N2V各运行两套同规则分割：

1. 全局Otsu。
2. 局部Gaussian阈值，window51；阈值为局部均值+1.5×原图估计sigma。

均仅删除小于16px连通域，未进行桥接、填洞或针对某张图调阈值。六份完整100层二值输出均为0/255。它们是固定基线候选，不是经过验证的ER识别器。

## 结果与细丝保真性

五张切片中，N2V使高频wavelet sigma代理量降低约5.4%–20.9%。NLM在该估计强度下变化很小，不能据此认为所有传统去噪都无效。N2V与原图Pearson相关约0.9958–0.9972，**高相关不代表ER结构正确**。

五张的平均选中面积：原图Otsu28.38%，NLM Otsu28.33%，N2V Otsu32.96%；原图local35.24%，NLM local35.24%，N2V local35.03%。

视觉检查显示：

- 全局Otsu主要选择亮组织块，无法分离细ER网。
- 局部阈值将大量背景起伏和组织纹理纳入mask；N2V未解决这一问题。
- 放大后的残差图包含亮轮廓和团块，而不是仅有随机噪点。N2V改变了部分结构的强度/对比度；这不直接证明真实ER被删，但也不支持“只去噪、不影响结构”。
- 相对原图local候选骨架，约80.7%–86.3%在N2V骨架2px范围内；约9.1%–12.3%的N2V骨架距原图候选骨架超过2px。这是**相对候选结构变化**，不是生物学FN/FP，也不用于选模型。
- 没有可信完整人工ER标签，无法可靠判断某次删除、增加、连接是否正确，因此不报告Dice/IoU/Precision/Recall。

原始N2V存在棋盘伪影的已知风险，见[N2V2论文](https://arxiv.org/abs/2211.08512)。本实验未通过针对性频谱对照将可见纹理归因为该伪影，不能直接作此结论。

## 结论和下一步

**保留本轮为可复现实验，不升级生产流程。** N2V略减小了某些高频变化，但没有证据证明恢复了真实ER细丝；它也不能提供Fluo中原本缺失的ER分子特异性。简单Otsu/local分割不适合作为本批ER训练标签。

最有信息价值的后续工作是：确认源图是否已被重采样/平滑及可用原始采集；在可信ER阳性对应区域检验线条保真性，而不是继续整体降低阈值。若噪声独立性可支持，再用同样的样本隔离方案做一个N2V2/相关噪声对照，但本轮未自动开启第二次训练。现有ER阳性对应Fluo的部分监督方案应继续保持“未标注=未知”，不要以这次候选mask自动补全。

## 文件

输出目录：`outputs/er_n2v_20261029/`

- `N2V_denoised.tif`、`NLM_denoised.tif`：float32、100×1200×1200、明确ZYX轴。
- `raw_otsu_mask.tif`、`raw_local_mask.tif`、`NLM_otsu_mask.tif`、`NLM_local_mask.tif`、`N2V_otsu_mask.tif`、`N2V_local_mask.tif`：完整候选mask。
- `five_slice_comparison.png`、`six_method_masks.png`、`five_detail_residuals.png`、`validation_curve.png`。
- `preview_z001/025/050/075/100`：原图TIFF和高清对照。
- `audit.json`、`config.json`、`environment.txt`、`history.jsonl`、`training_result.json`、`diagnostics.json`、`structure_change_diagnostics.json`、`verification.json`：数据/参数/结果/校验。
- 最佳checkpoint：`training/checkpoints/er_fluo_n2v_2d_0/er_fluo_n2v_2d_epoch=11_step_step=360_val_loss=0.0111.ckpt`；SHA256 `5d5464abfa0a1b445f18e80321a0837a1642db9f013276c6b9e2022ed656c248`。

全部8份输出堆栈逐页检查尺寸、finite值、mask uint8 0/255及SHA256通过；目标源TIFF再次校验未改动。代码：`code/er_n2v_experiment.py`、`code/er_n2v_review.py`；协议：`docs/er_n2v_protocol_20261029.md`。
