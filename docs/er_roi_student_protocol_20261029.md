# ER局部ROI Fluo-only U-Net探索试验

## 数据和任务

使用 `outputs/er_expanded_roi_candidates_20261029` 新增40组320×320配对小块；保留原5个示例，不混入重复示例。32训练来自23298/23299/23961/23962，8验证来自23965。不同Z可能仍是同一细胞，不计作独立生物样本。23965以前属于历史训练集且已经查看，本轮只做重新初始化模型的探索性验证，不声称独立测试。23300/23963不参与本实验。

输入仅Fluo。1=已有ER伪标签阳性，0=ROI内低ER信号候选阴性，255=忽略。候选阴性未经人工逐像素确认，ER低强度也不证明无ER。只在有效像素上计算BCE和Dice；既不能把ROI填白，也不能把所有未标记区域当背景。

## 固定试验

- 现有2D U-Net，base filters16，保留原BatchNorm和dropout，不修改历史模型。
- 随机初始化；seed20261029；每块独立Fluo p1–p99.8归一化，预测必须一致。
- batch4；每epoch完整遍历32训练块（8steps）；配对90度旋转/左右翻转。
- Adam 0.0003；最多40epochs；验证masked loss最小选择checkpoint，patience8。
- 固定0.5阈值。记录训练/验证已知像素Dice、IoU、precision、recall，及忽略区预测比例。
- 保存所有40块概率图，显示Fluo、伪标签和预测。没有完整人工GT，不报告真实全图ER准确率。
- 本轮是改变监督区域后的可行性试验；不同于旧实验划分和归一化，不能据此声称严格单因素提升。

## 执行与成本

入口 `code/er_roi_student.py`；旧脚本保持不变。训练必须恰好一张GPU，并先执行GPU矩阵乘法。CPU仅做数据检查和单元测试，不训练。

训练包 `outputs/er_roi_student_ready_20261029`，仅40个Fluo+label NPZ和manifest。没有ER源通道、原始整栈、credentials。部署前还需恢复可用Runpod控制连接、核查实时价格和余额，制定限时终止/下载保护；当前没有创建Pod、上传或训练。

运行：

```sh
python code/er_roi_student.py verify --data outputs/er_roi_student_ready_20261029
python code/er_roi_student.py train --data outputs/er_roi_student_ready_20261029 --output outputs/er_roi_student_run_<new_timestamp>
```

结果必须先下载并校验checkpoint/manifest，再删除Pod。不要自动再训或根据展示图改阈值。
