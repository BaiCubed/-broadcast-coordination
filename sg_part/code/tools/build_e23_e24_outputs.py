from __future__ import annotations

import csv
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches as PptInches


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/outputs"
E23 = ROOT / "results/E23"
E24 = ROOT / "results/E24"

E23_FIGURES = [
    ("e23_relative_effect_curves", "E23：连续压力下各算法实际效果", "纵轴为 IEEE-69 网络约束后的可交付弃电降低率。"),
    ("e23_relative_advantage_curves", "E23：EPS 相对性能差值", "纵轴为 EPS 减去参照算法的效果差值，阴影为配对 bootstrap 区间。"),
    ("e23_physical_violation_curves", "E23：物理约束风险曲线", "纵轴为安全裁剪前相对无控制基线新增的物理违规请求比例。"),
    ("e23_boundary_summary", "E23：相对性能与物理风险边界", "纵轴为连续两个压力点满足边界判据时的首次压力位置。"),
    ("e23_dataset_boundary_distribution", "E23：不同数据集的相对性能边界", "纵轴为数据集名称，点表示各数据集单独计算的边界位置。"),
    ("imbalance_metric_summary", "E23：空间不平衡指标审计", "纵轴为设备数量、容量、类型和网络耦合不平衡指标。"),
    ("effect_vs_density_imbalance", "E23：设备密度不均衡与算法效果", "横轴为设备密度不均衡指标 I_N，纵轴为可交付弃电降低率。"),
    ("effect_vs_type_heterogeneity", "E23：数据类型异质性与算法效果", "横轴为区域类型异质性指标 I_T，纵轴为可交付弃电降低率。"),
    ("safety_vs_headroom_coupling", "E23：网络裕度耦合与算法效果", "横轴为设备密度与网络裕度耦合指标 I_H，纵轴为可交付弃电降低率。"),
    ("spatial_heterogeneity_algorithm_comparison", "E23：空间异质性条件下的算法对比", "横轴为 H0、H1、H2（25%、50%、75%）、H3、H4、H5，纵轴为可交付弃电降低率。"),
    ("region_density_assignments", "E23：IEEE-69 区域设备分配审计", "横轴为实验条件，纵轴为各 IEEE-69 区域的设备比例。"),
    ("e23_r2_pressure_curves", "E23：连续压力下的响应跟踪 R²", "三个面板分别显示请求强度、线路容量压力和远端集中度下的 R²。"),
    ("e23_r2_spatial_conditions", "E23：H0-H5 空间条件下的响应跟踪 R²", "横轴为空间条件，纵轴为实际吸收曲线相对弃电目标曲线的 R²。"),
    ("e23_reduction_vs_r2_spatial", "E23：弃电消纳效果与响应跟踪能力", "横轴为可交付弃电降低率，纵轴为响应跟踪 R²。"),
]

E24_FIGURES = [
    ("e24_ieee123_effect_comparison", "E24：IEEE-123 各算法效果", "四个面板分别对应均匀部署、末端馈线集中、末端节点集中和容量降额场景；纵轴为可交付弃电降低率。"),
    ("e24_ieee123_safety_audit", "E24：IEEE-123 网络安全审计", "红柱为新增违规请求比例，蓝柱为网络接受率，纵轴为百分比。"),
    ("e24_r2_scenario_comparison", "E24：IEEE-123 场景下的响应跟踪 R²", "横轴为 IEEE-123 网络场景，纵轴为实际吸收曲线的响应跟踪 R²。"),
]

SPATIAL_OVERVIEW = r'''## E23 空间实验条件总览

下面的 H0-H5 是 IEEE-69 空间异质性实验的完整条件。每个条件都使用 5000 台设备、17 个数据集、相同的设备 profile、相同的网络拓扑和 30 个 paired seeds。条件之间只改变设备类型和设备密度在 IEEE-69 八个区域中的空间分配。

|条件|核心变化|区域分配方式|用于回答的问题|
|-|-|-|-|
|H0|完全混合基准|每个区域保持相同的数据类型比例；设备数量按区域基础负荷比例分配。|在空间均匀条件下，各算法的基准效果和网络风险是什么。|
|H1|类型分区|设备数量仍按区域基础负荷比例分配；住宅、建筑、热负荷、配网/AMI、DER、充电分别偏向不同区域。|仅改变数据类型的空间分布，观察区域行为差异的影响。|
|H2-25|轻度密度集中|各区域类型比例保持全局一致；密度权重混合强度 s=0.25。|仅改变设备数量空间集中，测试轻度不均衡。|
|H2-50|中度密度集中|各区域类型比例保持全局一致；密度权重混合强度 s=0.50。|测试中等设备集中造成的局部网络压力。|
|H2-75|高度密度集中|各区域类型比例保持全局一致；密度权重混合强度 s=0.75。|测试严重设备集中造成的局部瓶颈。|
|H3|类型与密度联合不均衡|在 H2-50 的密度集中上加入类型分区：远端低裕度区域偏住宅、热负荷和充电，中部偏建筑和 DER，近端高裕度区域偏住宅和 AMI。|测试设备数量差异和数据类型差异叠加后的效果。|
|H4|不利网络耦合|沿用 H3 的设备密度和全局类型配额，把高功率类型放入网络裕度最低区域。|测试高功率响应与网络瓶颈重合时的最不利情况。|
|H5|有利网络耦合|沿用 H3 的设备密度和全局类型配额，把高功率类型放入网络裕度较高区域。|作为 H4 的配对对照，测试有利空间匹配。|

### 条件之间的比较关系

- H0 → H1：只改变数据类型的区域分布，设备密度保持基准分配。
- H0 → H2-25/H2-50/H2-75：只改变设备密度集中程度，数据类型比例保持全局一致。
- H1 + H2-50 → H3：把类型分区和中度密度集中叠加，观察联合效应。
- H3 → H4/H5：设备数量、全局数据集比例和密度骨架保持一致，只改变高功率类型与网络裕度的匹配方向。
- H4 与 H5 是最重要的配对比较：H4 将高功率类型放在低裕度区域，H5 将其放在高裕度区域，二者效果差异直接反映空间位置的影响。

图中点形与条件对应关系为：圆形 H0、方形 H1、三角形 H2-25、菱形 H2-50、五边形 H2-75、叉形 H3、倒三角 H4、左三角 H5；颜色始终表示算法。

### H2 密度集中度的计算

设 IEEE-69 八个区域的基础负荷权重为 `q_r`，满足 `sum(q_r)=1`；令 `L` 为网络裕度最低的三分之一区域。先构造只把密度偏向 `L` 的目标权重：

`q_r^low = q_r / sum(q_j, j in L)`（当 `r in L`），其余区域为 0。

H2 的实际区域权重不是直接把 25%、50% 或 75% 的设备硬塞进远端，而是按集中强度 `s` 在基准分配和低裕度偏置分配之间插值：

`q_r(s) = (1-s) q_r + s q_r^low`。

因此 H2-25、H2-50、H2-75 分别使用 `s=0.25、0.50、0.75`。随后按照 `round(5000 q_r(s))` 分配区域设备数量，并用整数分配保证总数仍为 5000。H2 各区域内部继续使用全局数据集比例，因此 H2 单独改变的是设备密度，不改变类型组成。

对应的数量不平衡指标为：

`I_N = sqrt(sum_r q_r (x_r/q_r - 1)^2)`，其中 `x_r` 是实际区域设备占比。

`I_N=0` 表示设备数量严格遵循基础负荷权重；`I_N` 越大表示区域设备密度越偏离基础负荷分配。当前 30 个 seed 的典型值为 H2-25≈0.176、H2-50≈0.352、H2-75≈0.528，这些数值是实际分配后的审计结果，不是另外人为指定的横坐标。
'''

R2_INLINE = r'''## E23 R² 图片说明

R² 使用与弃电率降低率相同的逐条件结果。E23 连续压力部分以 dataset-seed-condition-algorithm 为配对单位；E23 空间异质性部分以包含 17 个数据集的 fleet-seed-condition-algorithm 为配对单位；E24 以 dataset-seed-scenario-algorithm 为配对单位。每个时间步的目标 `C_t` 是该时间步可用的弃电量，实际值 `A_t` 是经过设备 SOC、功率和 IEEE 网络安全层后真正吸收的弃电量。计算公式为：

`R² = 1 - sum_t(A_t-C_t)^2 / sum_t(C_t-mean(C))^2`。

R²=1 表示实际吸收曲线完全跟随弃电目标曲线；R²=0 表示不优于用目标均值作常数预测；R²<0 表示时间形状偏差大于该常数基准。R² 衡量时间跟踪能力，不能替代可交付弃电降低率这一总量指标。

## 12. E23 连续压力下的响应跟踪 R²

![连续压力下的响应跟踪 R²](figures/e23_r2_pressure_curves.png)

### 这张图回答什么
它回答：请求强度增加、线路容量下降或设备向远端集中时，各算法是否仍能跟随每个时间步的弃电变化。

### 横纵坐标
- 三个面板横坐标分别是请求强度 `q`、线路容量压力 `lambda_line` 和远端空间集中度 `kappa`，定义与 E23 连续压力图一致。
- 纵坐标是响应跟踪 R²；虚线是 R²=0.95 的参考线。

### 图中元素
- 每条曲线是一种算法，阴影是按相同 dataset-seed 配对观测进行 bootstrap 得到的 95% 区间。
- Centralized greedy UB 仍是设备层理论上界；它的 R² 只作参考，不能解读为网络安全控制性能。

### 结果和含义
该图与弃电率降低率曲线互补：某算法可能保持较高总量消纳，却因网络裁剪或 SOC 饱和而出现较低 R²。线路容量和空间集中压力下，EPS 的全局广播请求更容易被局部安全层裁剪，因此应同时观察 R²、可交付弃电降低率和网络接受率。

## 13. E23 H0-H5 空间条件下的响应跟踪 R²

![H0-H5 空间条件下的响应跟踪 R²](figures/e23_r2_spatial_conditions.png)

### 这张图回答什么
它回答：类型分区、密度集中和网络裕度耦合是否改变算法跟踪弃电时间变化的能力。

### 横纵坐标
- 横坐标为 H0、H1、H2-25、H2-50、H2-75、H3、H4、H5。
- 纵坐标为 R²；每个点为该条件下 17 个数据集和 30 个 paired seeds 的平均值，阴影为 95% bootstrap 区间。

### 图中元素
- 颜色表示算法，线条表示同一算法跨空间条件的变化趋势。
- H2-25、H2-50、H2-75 表示密度权重混合强度 `s=0.25、0.50、0.75`，不是三个新的数据集。
- R²=0.95 虚线用于识别高跟踪精度区域。

### 结果和含义
H4/H5 的 R² 差异可用于判断高功率类型放置位置是否影响时间跟踪，而不仅是影响总吸收量。若 EPS 的弃电率降低率保持较高但 R² 下降，说明它能吸收较多总量，却不能稳定跟随局部时间变化；这正是全局广播缺少区域状态信息的表现。

## 14. E23 弃电消纳效果与响应跟踪能力

![弃电消纳效果与响应跟踪能力](figures/e23_reduction_vs_r2_spatial.png)

### 这张图回答什么
它回答：算法是否同时具备高总量消纳效果和高时间跟踪能力。

### 横纵坐标
- 横坐标是可交付弃电降低率（%）。
- 纵坐标是响应跟踪 R²。
- 每个点是一个算法与空间条件的组合；30 个 seed 的结果先在条件内取均值。

### 图中元素
- 颜色表示算法，点形表示 H0-H5。
- 右上区域表示总量效果和时间跟踪都好；右下表示总量效果高但时间跟踪差；左上表示跟踪形状好但总量吸收不足。
- R²=0.95 虚线是时间跟踪参考线，不是弃电率降低率的判定线。

### 结果和含义
该图避免把 R² 单独当成“算法效果”。例如实际吸收曲线若始终为目标的 80%，曲线形状可能仍高度一致，R² 接近 0.96，但总量效果只有 80%。因此 EPS 与 baseline 的评价必须同时看两个坐标以及网络接受率。

## 15. E24 IEEE-123 场景下的响应跟踪 R²

![IEEE-123 场景下的响应跟踪 R²](../E24/figures/e24_r2_scenario_comparison.png)

### 这张图回答什么
它回答：从均匀部署到末端集中和容量降额时，IEEE-123 网络规模外推是否改变各算法的时间跟踪能力。

### 横纵坐标
- 横坐标为 S0 均匀部署、S1 末端馈线 50% 集中、S2 末端节点 80% 集中、S3 集中且容量降至 70%。
- 纵坐标为响应跟踪 R²；虚线为 R²=0.95。

### 图中元素
- 每条曲线是一种算法，阴影为 3 个代表数据集和 10 个 seed 的 bootstrap 区间。
- E24 的 EPS 使用 IEEE-69 直接训练冻结模型，因此该图是 IEEE-123 规模外推审计，不是 IEEE-123 直接训练结果。

### 结果和含义
如果 S2/S3 的 R² 显著下降，说明末端空间瓶颈和容量降额不仅减少总量消纳，也改变实际响应曲线的时间形状。将该图与 E24 弃电率降低率和网络安全审计图并列，可区分“吸收量减少”和“跟踪能力下降”两种影响。
'''

SUPPLEMENT = r'''
## 补充一：H0-H5 场景定义与构造方式

### 1. E23 空间异质性条件的真实定义

E23 空间异质性实验使用 IEEE-69 的 8 个区域：R1=2--27、R2=28--35、R3=36--46、R4=47--50、R5=51--52、R6=53--65、R7=66--67、R8=68--69。总设备数固定为 5000 台，17 个数据集的全局配额固定，30 个 seed 在所有条件和算法之间复用同一批设备 profile。实验改变的是设备和数据类型如何放入不同区域。

空间异质性部分使用 E21 的 `e21_mixed_aggregate` aggregate EPS controller；连续压力部分使用 IEEE-69 直接训练的 EPS fused controller。两部分均固定 IEEE-69 网络、设备总量、全局数据集配额和配对 seed，区别在于空间异质性部分改变设备在 R1-R8 之间的分配方式。

|编号|实际含义|具体构造|保持不变的内容|
|-|-|-|-|
|H0|区域完全混合|每个区域采用相同的数据类型比例，设备数量按区域基础负荷比例分配。|5000 台设备、17 个数据集全局配额、设备参数和输入场景。|
|H1|数据类型空间分区|区域设备数量仍按基础负荷权重分配，但区域偏好不同类型：R1-R2 偏住宅，R3 偏建筑，R4 偏热负荷，R5-R6 偏配网/AMI，R7 偏 DER，R8 偏充电。偏好权重为 8，其余类型权重为 1；随后用 IPF 同时恢复区域总量和数据集总量。|区域设备总量、全局数据集配额、设备数量和网络条件。|
|H2|密度分区|各区域数据类型比例保持全局一致，设备密度设置为 25%、50%、75% 三个集中强度。|数据类型组成、总设备数和全局数据集比例。|
|H3|类型与密度联合不均衡|在 50% 密度集中基础上叠加类型分区：低裕度远端区域偏住宅、热负荷和充电，中部区域偏建筑和 DER，近端高裕度区域偏低功率住宅和 AMI。|总设备数、全局数据集比例和设备参数。|
|H4|不利网络耦合|保持 H3 的区域设备密度和全局类型比例，将高密度、高功率类型放到网络裕度最低区域。|设备总量、区域密度和全局类型比例。|
|H5|有利网络耦合|保持 H3 的区域设备密度和全局类型比例，将高密度、高功率类型放到网络裕度较高区域。|设备总量、区域密度和全局类型比例。|

H0 的“完全混合”表示每个区域包含相同的数据类型比例，设备数量仍按区域基础负荷比例分配。H4 和 H5 共用 H3 的区域设备数量和全局类型比例，只有高功率类型与网络裕度的匹配方向不同。

### 2. H2 的三个密度强度

H2 使用三个强度审计设备密度效应：25% 表示轻度集中，50% 表示中度集中，75% 表示高度集中。三种强度均保持每个区域的数据类型比例与全局比例一致。

## 补充二：指标到底表示什么

- **可交付弃电降低率**：经过设备 SOC、功率、线路、节点电压和主变安全层之后，实际吸收的弃电量除以无控制时的弃电量。它衡量最终实现的效果，原始请求量作为独立的安全审计指标记录。
- **网络接受率**：安全层实际接受的控制功率除以算法原始请求功率。100% 表示请求基本全部通过；较低表示安全层对请求进行了大量裁剪。
- **新增违规请求比例**：在安全裁剪之前，算法请求相对于无控制基线新增线路过载、电压越限或主变过载的时间比例。它反映请求是否天然适配网络，不等同于最终执行后的违规率。
- **I_N**：区域设备数量密度不均衡指标，0 表示按基准区域权重分配；越大表示设备越集中。
- **I_C**：区域可控容量密度不均衡指标。它与 I_N 使用相同的加权离散度形式，输入量替换为设备容量。
- **I_T**：区域数据类型分布与全局数据类型分布之间的加权 Jensen-Shannon divergence。越大表示区域的住宅、建筑、热负荷、AMI、DER、充电组成越不同。
- **I_H**：设备密度与低网络裕度的相关性。正值表示更多设备位于低裕度区域；I_CH 则把设备数量换成可控容量，用于识别高功率设备是否集中在瓶颈附近。

Centralized greedy UB 只计算设备层理论上最多可以吸收多少弃电，不能作为经过 IEEE 潮流安全校核的可执行算法，也不能用它证明网络安全。

## 补充三：E23 连续线路容量实验的详细结果

线路容量压力轴的结果如下。数值为 17 个数据集和 30 个 paired seed 的均值；效果单位为百分比，接受率为原始请求的通过比例。

|算法|lambda_line=0 效果|lambda_line=0.4 效果|效果变化|接受率 0|接受率 0.4|新增违规率 0.4|
|-|-|-|-|-|-|-|
|EPS IEEE-69 fused|89.4|77.0|-12.4 个百分点|74.4|44.4|32.7|
|MPC|95.7|95.7|约 0|100.0|100.0|0.4|
|Mean-field control|74.1|74.2|约 0|97.5|97.5|1.9|
|Virtual battery|79.9|79.9|约 0|100.0|100.0|0.4|
|Packetized Energy Management|90.4|90.3|约 0|100.0|99.9|0.5|
|Transactive control|93.1|93.1|约 0|99.9|99.9|1.2|
|Centralized greedy UB|98.7|98.7|约 0|100.0|100.0|0.4|

### 为什么线路降额主要影响 EPS

线路容量对不同算法的影响取决于各算法的**请求生成方式**。

1. EPS 使用 IEEE-69 直接训练的 fused controller。它接收广播统计量并输出面向整个设备群的统一强度；模型输入不包含每条线路的实时剩余容量、设备所在区域或节点电压。它在名义容量下学到的请求偏向于尽可能多地吸收弃电，因此原始请求较激进。
2. 线路容量从额定值降到 60% 后，EPS 仍按同一类全局广播策略发出较大的请求。网络安全层只能在请求发出之后把低裕度支路上的功率裁掉，所以 EPS 的网络接受率从 74.4% 降到 44.4%，实际可交付效果从 89.4% 降到 77.0%，新增违规请求比例从 21.9% 升到 32.7%。
3. MPC、Virtual battery 和 PEM 在这组压力范围内生成的请求量或空间分布没有把新增容量压力推到可执行动作的瓶颈上。它们的网络接受率接近 100%，因此容量降额没有显著改变最终效果。Mean-field 和 Transactive 也只发生很小变化，说明它们在该实验的请求规模下仍有较大的网络余量。
4. Local SOC rules 的接受率本来就很低，约 16%，新增线路降额无法再显著降低它的效果；其低效果主要来自本地规则和设备 SOC，线路容量只贡献了较小的边际变化。

在当前 IEEE-69、容量降额范围和请求规模下，其他方法的请求尚未成为网络瓶颈。EPS 的结果给出了一个明确边界：**只使用全局广播、没有线路余量输入的控制器，在网络容量变化时需要额外的容量感知或安全反馈机制。** 当前 EPS 的闭环短缺修正只能根据实际吸收结果调整广播强度，无法预先定位即将饱和的支路，也就无法提前重新分配请求。

## 补充四：E23 空间异质性实验的结果与原因

|条件|EPS 效果|最佳可执行 baseline|EPS 减 baseline|EPS 接受率|EPS 新增违规率|主要解释|
|-|-|-|-|-|-|-|
|H0|90.9|MPC 94.1|-3.2|72.3|34.9|均衡基准下 EPS 已有较高效果，但全局广播仍有请求过量和空间裁剪。|
|H1|87.0|MPC 84.8|+2.2|69.5|36.0|类型分区改变区域行为，但设备密度仍按负荷权重分配；EPS 的平均效果下降约 3.9 个百分点。|
|H2-25|91.4|MPC 94.7|-3.3|71.6|35.9|轻度密度集中对 EPS 的平均效果影响较小，但低裕度区域的请求裁剪增加。|
|H2-50|91.1|MPC 94.7|-3.6|69.1|36.8|中度密度集中后，EPS 接受率继续偏低；MPC 通过网络校核保持较高可交付效果。|
|H2-75|91.4|MPC 95.4|-4.0|63.9|38.5|高度密度集中提高局部支路压力，EPS 接受率进一步下降。|
|H3|91.2|MPC 92.2|-1.0|69.6|36.4|类型分区和密度集中叠加后，EPS 与 MPC 的差距缩小，网络裁剪仍然明显。|
|H4|90.0|MPC 93.0|-2.9|69.1|37.1|高功率类型与低裕度区域重合，EPS 的接受率和效果下降。|
|H5|92.1|MPC 94.5|-2.4|69.6|35.8|高功率类型与高裕度区域匹配，EPS 比 H4 高约 2.1 个百分点。|

H4-H5 的配对差异来自高功率类型与网络裕度的空间匹配：两种条件使用相同的设备数量、区域密度和全局数据集配额，仅改变类型映射。H4 比 H5 低约 2.1 个百分点，表明高功率设备所在支路会直接改变可交付效果。

## 补充五：E24 IEEE-123 结果与 E23 的关系

|场景|EPS 效果|最佳可执行 baseline 效果|EPS 接受率|EPS 新增违规率|含义|
|-|-|-|-|-|-|
|S0 均匀部署|96.7|Transactive 93.4|60.2|24.0|网络扩大但均匀部署下仍有较高设备层效果，EPS 原始请求已经需要较多安全裁剪。|
|S1 50% 末端馈线集中|86.1|Transactive 84.5|29.1|37.0|空间集中使馈线成为瓶颈，EPS 的接受率明显下降。|
|S2 80% 末端节点集中|39.4|Virtual battery 22.1|7.6|38.9|所有方法都受到局部节点瓶颈；EPS 的效果仍较高主要因为它发出的请求更多，不代表它更安全或更高效。|
|S3 50% 末端馈线集中且容量降至 70%|69.2|MPC 78.5|17.5|36.3|容量进一步下降后 EPS 落后于网络感知 baseline，说明规模外推时全局广播的安全边界更早显现。|

E24 与 E23 的线路容量结论需要区分：E23 的连续扫描显示 MPC、PEM 等 baseline 在该范围内尚未达到请求瓶颈，因此曲线近似水平；E24 的 S3 同时包含末端集中和容量降额，baseline 也开始被网络限制，所以 MPC 从 S1 的 80.0% 降到 S3 的 78.5%，PEM 从 76.1% 降到 69.7%。EPS 则从 S1 的 86.1% 降到 S3 的 69.2%，下降更明显。

E24 的主要原因仍是控制器与网络信息的匹配程度：EPS 使用 IEEE-69 冻结模型，未在 IEEE-123 上直接训练，也没有把 IEEE-123 的节点、支路余量和调压器状态作为广播输入。安全层可以阻止不安全功率真正执行，但它只能事后裁剪，不能让 EPS 在发送前把请求转移到有余量的位置。MPC、Mean-field、Virtual battery 等 baseline 在此实现中更保守或直接使用网络校核，因此在 S3 的接受率更高；Transactive 在 S0/S1 的效果较好，但空间集中后其新增违规率也明显增加。

## 补充六：应如何解释实验结论

本组实验支持的结论是：设备规模带来的统计聚合效果并不能自动消除配电网的空间约束；当设备集中到低裕度区域，或者网络容量降低时，全局广播 EPS 可能先出现请求裁剪和相对性能下降。它同时给出了 EPS 的工程有效边界：在当前 IEEE-69/IEEE-123 构造、设备参数、容量压力和安全层定义下，EPS 的效果和安全性开始受容量与空间信息缺失影响。

结论范围限定为当前 IEEE-69/IEEE-123 构造、设备参数、容量压力和安全层定义。实验结果不外推到任意线路容量、IEEE-123 直接训练、完整三相潮流或现场运行；E24 使用简化单相网络，E23/E24 均属于可复现仿真审计，模型、网络参数和未建模运行控制设备仍构成结果边界。

## 补充七：算法角色与差异来源

|算法|在线复杂度|接收的信息和控制方式|结果中的典型特点|
|-|-|-|-|
|No coordination|O(0)|不发送控制请求|零效果参照，同时用于计算新增网络风险。|
|Local SOC rules|O(1) per device|每台设备只使用本地 SOC、可用性和时段规则|通信量低，行为保守，平均网络接受率低，效果受设备本地状态限制。|
|MPC|O(HN)|使用预测窗口和设备/网络约束进行滚动分配|E23 空间条件下通常是最佳可执行 baseline，但计算量随设备数和预测窗口增长。|
|Mean-field control|O(N)|使用群体状态分布计算参与概率|能保持较高接受率，但在当前参数下可交付效果低于 MPC、PEM 和 Transactive。|
|Virtual battery|O(N)|把设备群压缩为等效电池并分配总功率|容量聚合稳定，空间差异较小，局部网络信息有限。|
|Packetized Energy Management|O(N log N)|按功率包排序、接纳和分配|设备层效果较高，接受率通常接近100%，需要排序和逐设备处理。|
|Transactive control|O(N log N)|根据设备报价或优先级排序并清算|均匀场景效果较高；空间集中和容量收紧后，接受率和效果下降更明显。|
|EPS|O(1)|根据广播统计量输出一个群体级控制强度，并进行融合与闭环修正|通信开销最低，平均效果较高；模型没有线路余量、区域位置和节点电压输入，网络压力下容易产生过量请求。|
|Centralized greedy UB|O(N)|计算设备层理论可吸收上限|接近100%的理论参照，不计入可执行 baseline 排名。|

EPS 的优势来自群体级广播：控制器只需要发送一个全局信号，在线通信和调度复杂度保持 O(1)。它的工程代价也直接来自这个信息边界：单个广播强度无法区分低裕度远端支路和高裕度区域，网络安全层需要在请求生成后进行局部裁剪。

## 补充八：E23 IEEE-69 空间条件的实际结果

以下数值来自 30 个 seed 的 `spatial_heterogeneity_summary.csv`，单位为可交付弃电降低率百分比。Centralized greedy UB 单独作为理论上界；“可执行最佳”只在 MPC、Mean-field control、Virtual battery、PEM 和 Transactive control 中取最大值。

|条件|Local SOC|MPC|Mean-field|Virtual battery|PEM|Transactive|EPS|可执行最佳|EPS网络接受率|EPS新增违规|
|-|-|-|-|-|-|-|-|-|-|-|
|H0|67.6|94.1|75.3|79.9|90.0|85.1|90.9|94.1|72.3|34.9|
|H1|63.9|84.8|71.1|74.8|81.9|81.6|87.0|84.8|69.5|36.0|
|H2-25|66.7|94.7|75.4|79.9|90.3|85.3|91.4|94.7|71.6|35.9|
|H2-50|51.7|94.7|75.6|80.0|90.4|85.2|91.1|94.7|69.1|36.8|
|H2-75|59.4|95.4|76.0|80.1|91.1|85.6|91.4|95.4|63.9|38.5|
|H3|53.5|92.2|73.3|76.8|87.4|84.8|91.2|92.2|69.6|36.4|
|H4|48.2|93.0|75.5|80.0|89.7|84.9|90.0|93.0|69.1|37.1|
|H5|52.2|94.5|75.1|79.3|89.7|85.0|92.1|94.5|69.6|35.8|

### 结果分析

- H0 中 EPS 达到 90.9%，低于 MPC 的 94.1%，接近 PEM 的 90.0%，明显高于 Local SOC、Mean-field 和 Virtual battery。此结果说明 EPS 在混合场景下具备较高设备群体利用效果，同时仍承担了全局广播缺少空间信息的网络裁剪代价。
- H1 将不同数据类型放入不同区域后，EPS 从 H0 的 90.9% 降至 87.0%。区域行为差异改变了同一广播强度对应的实际功率响应；MPC 也从 94.1% 降至 84.8%，说明类型分区会影响所有控制器，EPS 的下降幅度相对较小。
- H2 的密度强度从 25% 提高到 75% 时，EPS 均值约为 91.4%、91.1% 和 91.4%，平均效果没有严格单调下降，但网络接受率从 71.6% 降至 63.9%，新增违规从 35.9% 升至 38.5%。局部瓶颈已经增加，平均弃电降低率被其他区域的可用容量部分抵消。
- H3 同时加入类型分区和50%密度集中，EPS 为 91.2%，MPC 为 92.2%。类型空间差异与设备密度叠加后，两者差距缩小到约1.0个百分点，说明 EPS 在该联合场景仍保持较强平均效果，但接受率只有69.6%。
- H4 与 H5 使用相同区域设备数量和全局数据集配额，只改变高功率类型与网络裕度的对应关系。EPS 在 H4 为90.0%，在 H5 为92.1%，相差2.1个百分点；这直接说明设备位置和设备类型共同决定可交付效果。

## 补充九：线路容量为什么主要影响 EPS

在 E23 的连续线路降额实验中，线路容量从额定值降至60%时，结果如下：

|算法|额定容量效果|容量60%效果|变化|额定容量接受率|容量60%接受率|
|-|-|-|-|-|-|
|EPS|89.4|77.0|-12.4|74.4|44.4|
|MPC|95.7|95.7|0.0|约100|约100|
|Mean-field control|74.1|74.2|+0.1|约100|约100|
|Virtual battery|79.9|79.9|0.0|约100|约100|
|PEM|90.4|90.3|-0.1|约100|99.9|
|Transactive control|93.1|93.1|0.0|约100|约100|
|Centralized greedy UB|98.7|98.7|0.0|约100|约100|

EPS 的接受率由74.4%降至44.4%，新增违规请求由21.9%升至32.7%。原因有三层：

1. EPS 的输入是群体广播统计量，当前输入没有每条线路的剩余容量、设备区域和节点电压。模型在额定网络上学习到的控制强度，在容量降额后仍会向所有区域发送相近的群体请求。
2. 安全层在请求生成后才执行支路、电压和主变约束。低裕度支路上的请求被裁剪，造成网络接受率下降，实际吸收的弃电减少，原始请求风险同时升高。
3. MPC、Virtual battery、PEM 和 Transactive 在本实验实现及压力范围内产生的请求更保守，或其设备/网络校核没有把线路容量推到瓶颈，因此最终效果近似不变。这个结果描述的是当前请求规模下的工作区间，不能推广为这些算法对线路容量普遍不敏感。

Local SOC rules 的效果本来较低，主要受本地 SOC 和可用性限制；它的接受率已经偏低，容量降额带来的额外变化有限。

因此，E23 对 EPS 的评价应同时包含两面：EPS 以 O(1) 通信取得较高平均效果，但在网络容量变化时暴露出全局广播的信息边界。增加线路余量或区域状态反馈可以提高安全性，但会改变当前 O(1) 信息假设，需要作为后续控制设计单独验证。

## 补充十：E24 IEEE-123 结果对比

当前 E24 结果使用 IEEE-69 直接训练的 EPS 冻结模型，作用是测试规模外推；下表用于分析现有 E24 输出，不代表 IEEE-123 直接训练结果。

|场景|Local SOC|MPC|Mean-field|Virtual battery|PEM|Transactive|EPS|EPS接受率|EPS新增违规|
|-|-|-|-|-|-|-|-|-|-|
|S0 均匀部署|59.1|84.8|69.6|74.5|81.9|93.4|96.7|60.2|24.0|
|S1 末端馈线50%集中|75.2|80.0|65.8|70.4|76.1|84.5|86.1|29.1|37.0|
|S2 末端节点80%集中|19.2|21.7|21.1|22.1|9.8|13.4|39.4|7.6|38.9|
|S3 集中且容量70%|64.3|78.5|65.4|70.0|69.7|79.1|69.2|17.5|36.3|

E24 中 EPS 在 S0、S1 的设备层效果最高，但接受率分别只有60.2%和29.1%。S2 的末端节点集中使所有可执行算法效果明显下降；EPS 仍有39.4%，主要来自其较大的原始请求量，接受率仅7.6%，不能将其解释为安全优势。S3 同时施加空间集中和容量降额后，MPC、Transactive 和其他 baseline 的结果接近或超过 EPS，说明网络约束成为主导因素。

## 补充十一：综合评价

EPS 的主要优势是低通信复杂度和较高群体平均消纳效果，尤其在 H0、H2 和 E24 的均匀部署场景中表现明显。它的主要弱点是控制器只输出群体级广播强度，无法根据线路余量和设备区域主动避开局部瓶颈，因此在 H4、容量降额、末端集中等条件下，网络接受率下降较快。

MPC 是当前 IEEE-69 空间实验中最稳定的可执行 baseline，但复杂度为 O(HN)。PEM 和 Transactive 在均匀场景中具有较高效果，空间集中后 Transactive 的接受率下降明显；Virtual battery 的效果较稳定，但平均水平低于 MPC 和 PEM；Mean-field control 的接受率较高，设备效果中等；Local SOC rules 通信开销低，但本地信息不足以形成高效群体调度。

总体结果支持的结论是：EPS 的 O(1) 广播机制在统计聚合场景中有效，但统计规模无法自动消除线路容量、电压、主变和空间集中带来的局部约束。EPS 的有效边界应同时用可交付弃电降低率、网络接受率和新增违规请求比例评价。

## 补充十二：完整实验范围与额外指标

H0-H5 只描述 E23 空间异质性这一部分。E23 和 E24 的完整实验还包括以下内容：

|实验部分|实际变化|固定内容|主要回答的问题|
|-|-|-|-|
|E23 连续请求压力|请求强度 q 从 0.0 扫描到 1.5，步长 0.1|IEEE-69 网络、17 个数据集、5000 台设备、30 个 paired seeds|EPS 在同时响应需求增强时何时相对 baseline 失去优势。|
|E23 连续线路容量压力|线路和主变容量从额定值降至 60%，压力步长为 0.05|请求场景、设备 profile、数据集和 seed|网络容量下降是否使 EPS 的全局广播请求变得不可执行。|
|E23 连续空间集中压力|0% 到 80% 的设备放置到 IEEE-69 远端节点，步长为 10%|总设备数、设备类型配额、算法和 seed|局部集中响应是否造成馈线和末端节点瓶颈。|
|E23 空间异质性|H0、H1、H2（25%、50%、75%）、H3、H4、H5|总设备数、全局数据集比例、设备参数和网络拓扑|数据类型、设备密度和网络裕度的空间组合如何改变效果。|
|E24 IEEE-123 审计|S0 均匀、S1 末端馈线 50%、S2 末端节点 80%、S3 末端馈线 50% 且容量 70%|IEEE-123 等值拓扑、3 个代表数据集、10 个 seeds|网络规模扩大和末端集中是否进一步压缩 EPS 的可交付效果。|

### 空间交互效应

空间实验还保存了类型分区和密度分区的联合效应，用于判断两种空间差异是否简单相加。对每种算法计算：

`J = R(H3) - R(H1) - R(H2-50) + R(H0)`

其中 `R` 是可交付弃电降低率。`J>0` 表示联合场景的损失小于两个单独因素损失的简单相加，`J<0` 表示出现额外的交互损失。当前结果中 EPS 的 `J` 约为 `+4.0` 个百分点，MPC 约为 `+6.9` 个百分点，说明 H3 的联合结果不能只用 H1 和 H2 的单独结果线性推断。

H4 与 H5 还提供网络耦合方向的配对差值 `R(H4)-R(H5)`。EPS 约为 `-2.1` 个百分点，表示高功率、高密度设备放入低裕度区域后，实际可交付效果下降。

### 容量耦合指标 I_CH

空间审计除 `I_N`、`I_C`、`I_T` 和 `I_H` 外，还计算 `I_CH`：可控容量密度与低网络裕度之间的 Pearson 相关系数。它用于区分“设备数量集中”与“高功率设备集中”两种情况。H4/H5 的设备数量密度相同，但 `I_CH` 的方向和数值反映高功率设备是否与网络瓶颈重合。当前结果中 H4/H5 的 `I_CH` 均约为 0.88，配对效果差异主要来自高功率类型的具体区域映射。

### 网络约束覆盖范围

E23 和 E24 的执行器在每个时间步检查线路容量、节点电压和变压器容量；控制请求先经过设备 SOC、功率和可用率约束，再经过网络安全层裁剪。结果同时保存：可交付效果、网络接受率、最低电压、最大线路负载、最大变压器负载，以及安全裁剪前相对无控制基线新增的违规请求比例。

当前实验没有模拟完整三相相不平衡、保护装置动作、调压器控制、故障状态、通信延迟和现场设备故障。因此实验可以回答“空间集中和容量压力如何影响控制器的可交付效果与请求风险”，不能替代完整配电网规划或现场运行认证。

### 数据集、模型和比较范围

E23 连续压力和空间异质性均使用 17 个数据集、5000 台设备和 30 个 paired seeds。连续压力部分使用 IEEE-69 直接训练的 EPS fused 模型；空间异质性部分使用 E21 aggregate EPS controller。E24 使用 LCL、HEAPO 和 EU-35297 三个代表数据集、10 个 seeds，并复用 IEEE-69 直接训练的 EPS 模型到 IEEE-123 单相结构等值上进行规模外推。

所有部分均比较 No coordination、Local SOC rules、MPC、Mean-field control、Virtual battery、PEM、Transactive control 和 EPS；Centralized greedy UB 只作为设备层理论上界，单独报告，不作为网络安全可执行算法排名。
'''


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


SPATIAL_INLINE = r'''# E23 空间异质性图片说明

空间异质性实验在 IEEE-69 的 8 个区域中重新分配相同的一组设备 profile。总设备数固定为 5000 台，17 个数据集的全局配额固定，30 个 seed 在所有条件和算法之间配对使用。空间实验使用 E21 的 `e21_mixed_aggregate` EPS controller；连续压力实验使用 IEEE-69 直接训练的 EPS fused controller。

实验条件包括 H0 区域完全混合、H1 数据类型分区、H2 密度分区（25%、50%、75%）、H3 类型与密度联合不均衡、H4 不利网络耦合和 H5 有利网络耦合。所有条件固定总设备数、全局数据集比例、设备容量、功率、SOC、可用率、网络拓扑和 seed，只改变设备到区域的映射。

## 6. IEEE-69 空间不平衡指标审计

![四个不平衡指标](figures/imbalance_metric_summary.png)

### 这张图回答什么
它回答：H0-H5 是否真的分别产生了设备密度、可控容量、数据类型和网络耦合差异。指标图是空间实验的构造审计，先确认实验条件确实改变了预期变量，再解释后面的算法效果。

### 横纵坐标
- 横坐标是 H0、H1、H2-25、H2-50、H2-75、H3、H4、H5。
- 纵坐标是无量纲不平衡指标；数值越大表示对应空间差异越强。
- `I_N` 衡量设备数量相对区域基础负荷权重的离散程度；`I_C` 衡量可控容量的离散程度；`I_T` 衡量区域类型组成与全局组成的差异；`I_H` 衡量设备密度与低网络裕度的相关性。

### 图中元素
- 每种颜色对应一个指标，柱高表示该条件下的指标值。
- H2 的三根柱分别表示轻度、中度和高度密度集中。
- `I_CH` 没有单独画在本图中，但保存在空间指标数据中，用于检查高功率设备容量与低裕度区域的耦合。

### 结果和含义
- H0 的 `I_N≈0.001`、`I_T≈0`，说明区域接近完全混合。
- H1 的 `I_T≈0.109`，说明类型分区是该条件的主要变化；设备数量密度仍接近基准。
- H2-25、H2-50、H2-75 的 `I_N` 约为 0.176、0.352、0.528，随集中强度递增，证明密度压力按设计增强。
- H3-H5 保持约 0.352 的数量不平衡，同时叠加类型差异和网络匹配差异，形成联合空间压力。
- `I_H` 在密度集中条件下约为 0.885，表示设备密度更多位于低裕度区域；H4/H5 的具体高功率类型位置还需要结合 `I_CH` 和算法效果配对分析。

## 7. 设备密度不平衡与算法效果

![密度不均衡与效果](figures/effect_vs_density_imbalance.png)

### 这张图回答什么
它回答：在数据类型组成保持相近时，单独增加设备数量集中是否会降低算法最终可交付的弃电消纳。

### 横纵坐标
- 横坐标是 `I_N`，表示区域设备数量相对基础负荷权重的偏离程度。
- 纵坐标是经过设备约束和 IEEE-69 网络安全层之后的可交付弃电降低率（%）。
- 每个点对应一个条件、一个 seed 和一种算法；颜色表示算法。

### 图中元素
- H0 和 H1 位于低 `I_N` 区域，H2-25、H2-50、H2-75 沿横轴逐步右移。
- EPS、MPC、Mean-field、Virtual battery、PEM、Transactive 和 Local SOC 使用同一坐标尺度比较。
- Centralized greedy UB 只表示设备层理论可吸收上限，不参与网络安全算法排名。
- 颜色表示算法；点形直接表示空间条件：圆形 H0、方形 H1、三角形 H2-25、菱形 H2-50、五边形 H2-75、叉形 H3、倒三角 H4、左三角 H5。图右下角的“点形：空间条件”图例给出完整对应关系。

### 结果和含义
EPS 在 H2-25、H2-50、H2-75 的平均效果约为 91.4%、91.1% 和 91.4%，平均效果没有严格单调下降；但对应网络接受率从 71.6% 降到 69.1% 和 63.9%，新增违规请求从 35.9% 增到 38.5%。这说明其他区域仍能提供可消纳容量，掩盖了局部拥塞对总效果的影响；接受率和违规率更早暴露了密度集中造成的网络压力。MPC 在这些条件下约为 94.7%-95.4%，说明网络感知的逐设备分配更能保持可交付效果。

## 8. 数据类型异质性与算法效果

![类型异质性与效果](figures/effect_vs_type_heterogeneity.png)

### 这张图回答什么
它回答：不同区域由住宅、建筑、热负荷、AMI、DER 和充电等不同类型主导时，数据类型的空间差异是否会改变控制效果。

### 横纵坐标
- 横坐标是 `I_T`，即区域类型分布与全局类型分布之间的加权 Jensen-Shannon divergence。
- `I_T=0` 表示各区域类型比例接近全局混合；数值越大表示类型分区越明显。
- 纵坐标是可交付弃电降低率（%）。

### 图中元素
- 每个点表示一个条件、seed 和算法的实际网络执行结果。
- H0、H2 的 `I_T` 接近 0，用作类型组成不变的参照；H1 具有最高的类型异质性；H3-H5 叠加密度和网络耦合。
- 点的颜色表示算法，便于观察同一类型异质性水平下不同算法的垂直差异。
- 点形直接标出 H 条件：圆形 H0、方形 H1、三角形 H2-25、菱形 H2-50、五边形 H2-75、叉形 H3、倒三角 H4、左三角 H5。因此 H0/H2 等横坐标接近的点仍可区分。

### 结果和含义
H0 的 EPS 效果为 90.9%，H1 降至 87.0%；MPC 从 94.1% 降至 84.8%。类型分区改变了相同广播强度对应的设备响应形状，使单一群体信号更难同时匹配不同区域。EPS 的下降幅度小于 MPC 在这组条件中的下降幅度，因此 H1 中 EPS 为 87.0%，高于 MPC 的 84.8%；这体现的是当前输入分布与模型匹配程度的差异，不代表 EPS 在所有类型分区场景都优于 MPC。

## 9. 网络裕度耦合与算法效果

![网络裕度耦合与效果](figures/safety_vs_headroom_coupling.png)

### 这张图回答什么
它回答：设备密度与网络低裕度区域的空间匹配，是否会改变最终效果。这里的网络裕度同时考虑支路热容量余量和区域最低电压余量。

### 横纵坐标
- 横坐标是 `I_H = corr(device density, 1-headroom)`。
- `I_H>0` 表示更多设备位于低裕度区域；`I_H<0` 表示设备更多位于高裕度区域。
- 纵坐标是可交付弃电降低率（%）。

### 图中元素
- 每个点是一个条件、seed 和算法的结果，颜色表示算法。
- H4 和 H5 具有相同的设备数量密度和全局数据集比例，主要变化是高功率类型与网络裕度的对应关系。
- `I_CH` 作为补充指标记录可控容量与低裕度的相关性，用来识别“设备数量相同但功率能力不同”的影响。
- 点形直接对应 H0-H5 条件，具体映射为圆形 H0、方形 H1、三角形 H2-25、菱形 H2-50、五边形 H2-75、叉形 H3、倒三角 H4、左三角 H5；颜色仍只表示算法。

### 结果和含义
EPS 在 H4 为 90.0%，在 H5 为 92.1%，H4-H5 相差约 2.1 个百分点；MPC 在 H4/H5 分别为 93.0% 和 94.5%。高功率设备放在低裕度区域时，统一广播请求更容易被安全层裁剪。H4/H5 的差异证明空间位置与设备类型的匹配会改变可交付效果，单看总设备数或全局数据集比例无法解释这一变化。

## 10. 不同空间条件下的算法对比

![不同空间条件下的算法对比](figures/spatial_heterogeneity_algorithm_comparison.png)

### 这张图回答什么
它回答：在相同设备规模和输入场景下，空间混合、类型分区、密度集中及网络耦合如何改变 EPS 与各 baseline 的相对表现。

### 横纵坐标
- 横坐标是 H0、H1、H2-25、H2-50、H2-75、H3、H4、H5。
- 纵坐标是可交付弃电降低率（%）。
- 每条曲线是一种算法；图例中的 `O(1)`、`O(N)`、`O(HN)` 等表示在线计算复杂度。

### 图中元素
- Local SOC rules 是低通信量的本地规则；MPC 使用预测窗口和约束分配；Mean-field control 使用群体状态分布；Virtual battery 使用等效电池；PEM 使用功率包接纳；Transactive control 使用报价或优先级清算。
- EPS 使用群体级广播强度，在线通信复杂度为 `O(1)`。
- Centralized greedy UB 是设备层理论上界，不作为可执行安全 baseline。

### 结果和含义
MPC 在 H0、H2 和 H4/H5 中通常是最佳可执行 baseline，约为 92.2%-95.4%；EPS 在这些条件中约为 90.0%-91.4%，保持较高平均效果但接受率较低。H1 中 EPS 为 87.0%，高于 MPC 的 84.8%，说明类型分区对不同控制器的影响取决于其输入信息和请求生成方式。Local SOC、Mean-field 和 Virtual battery 效果较低或更保守；PEM 和 Transactive 在均匀场景表现较好，但空间压力增大后优势减弱。整体结果表明 EPS 的主要优势是 `O(1)` 广播，代价是缺少区域线路余量和节点电压信息。

## 11. IEEE-69 区域设备分配审计

![区域密度构造检查](figures/region_density_assignments.png)

### 这张图回答什么
它回答：各空间条件是否按照预先定义的区域密度构造，且 H3、H4、H5 是否保持可配对比较。

### 横纵坐标
- 横坐标是实验条件 H0、H1、H2-25、H2-50、H2-75、H3、H4、H5。
- 纵坐标是各区域设备占总设备数的比例（%）。
- 每条线对应一个 IEEE-69 区域：R1=2--27、R2=28--35、R3=36--46、R4=47--50、R5=51--52、R6=53--65、R7=66--67、R8=68--69。

### 图中元素
- H0 的区域设备比例按区域基础负荷权重分配。
- H1 保持区域总量比例，同时改变区域内的数据类型偏好。
- H2-25、H2-50、H2-75 只改变密度集中强度，依次表示轻度、中度和高度集中。
- H3、H4、H5 使用相同的区域密度骨架；H4/H5 的配对变化来自高功率类型映射到低裕度或高裕度区域。

### 结果和含义
该图是实验设计的结构审计。它确认 H2 的三种强度确实形成逐步集中的区域分配，也确认 H4 和 H5 的设备密度保持一致。因此 H4-H5 的效果差异可以归因于类型与网络裕度的空间匹配，而不归因于总设备数变化。区域设备分配与 `I_N`、`I_C`、`I_T`、`I_H`、`I_CH` 指标共同构成空间实验的可复核记录。
'''


def _e23_description_source() -> str:
    root = _read(E23 / "FIGURE_DESCRIPTIONS.md")
    marker = "# E23 空间异质性图片说明"
    if marker in root:
        root = root.split(marker, 1)[0].rstrip()
    e23_r2 = R2_INLINE.split("## 15. E24", 1)[0].rstrip()
    return root + "\n\n" + SPATIAL_INLINE.strip() + "\n\n" + e23_r2


def _e24_description_source() -> str:
    e24_r2 = ("## 15. E24" + R2_INLINE.split("## 15. E24", 1)[1]).replace("../E24/figures/", "figures/")
    return _read(E24 / "FIGURE_DESCRIPTIONS.md") + "\n\n" + e24_r2


def _rewrite_links(text: str, result_dir: str) -> str:
    text = text.replace("](figures/", f"](../{result_dir}/figures/")
    text = text.replace("`figures/", f"`../{result_dir}/figures/")
    text = text.replace("`data/", f"`../{result_dir}/data/")
    return text


def _clean_document_language(text: str) -> str:
    replacements = {
        "E23只保留IEEE-69网络下的直接训练EPS fused和原有baseline，不包含迁移模型。每个压力点使用相同dataset、seed和设备样本进行配对比较。": "E23 连续压力实验使用 IEEE-69 直接训练的 EPS fused 和原有 baseline，每个压力点使用相同 dataset、seed 和设备样本进行配对比较。E23 空间异质性实验使用 E21 aggregate EPS controller，在相同的 IEEE-69 网络和配对设备样本上比较 H0-H5 的空间分布影响；两部分都不使用迁移到新网络的模型。",
        "，而不是只看算法发出了多少请求": "，并同时报告算法原始请求量",
        "，不是对所有电网都成立的数学普适边界": "，适用于本实验扫描范围和判据",
        "，不是两个独立样本的差异": "，体现相同输入下的配对差异",
        "，不是数学定理或电网法规限值": "，作为工程审计参考线，不替代数学定理或电网法规",
        "，不是数学定理。": "，用于工程审计参考。",
        "，而不是弃电降低率百分比": "，用于表示首次达到边界的压力参数",
        "，不是由理论预先指定的固定常数": "，由实验扫描范围和判据确定",
        "，而不是把所有数据集强行声称为同一个失效压力": "，用于报告数据集间的有效范围差异",
        "，不是完整多相OpenDSS潮流": "，执行范围不包含完整多相OpenDSS潮流",
        "，不是IEEE-123直接训练结果": "，执行范围不包含IEEE-123直接训练",
        "，不是算法未经校核的原始请求": "，同时排除算法未经校核的原始请求",
        "，不是数学定理，也不替代具体电网的运行标准": "，作为风险标尺，不替代具体电网的运行标准",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    metadata_prefixes = (
        "文件：", "文件:", "- PNG：", "- PDF：", "- 数据：", "- 生成脚本：",
        "生成脚本：", "- `figures/", "- `data/",
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(metadata_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines)


def _build_markdown() -> None:
    e23 = _e23_description_source()
    e24 = _e24_description_source()
    e23 = _rewrite_links(e23, "E23")
    e24 = _rewrite_links(e24, "E24")
    text = """# E23-E24：IEEE 配电网空间与规模约束实验汇总

本文件合并两个互补实验：E23 在 IEEE-69 上连续改变请求强度、线路容量压力、设备空间集中和数据类型空间分配，确定 EPS 的相对性能与物理风险边界；E24 将代表性场景外推到 IEEE-123 的简化单相结构等值，检查网络规模扩大后的效果和安全变化。

## 实验关系

- **E23：IEEE-69 连续压力与空间异质性。** 重点回答 EPS 在什么压力下相对 baseline 失去优势，以及设备集中、数据类型分区和网络裕度耦合如何改变效果。
- **E24：IEEE-123 规模外推审计。** 重点回答网络规模扩大、末端集中和容量降额是否进一步降低可交付效果，并增加网络安全风险。
- 两个实验都使用可交付弃电降低率作为主要效果指标，并同时报告安全层介入前的违规请求和安全层介入后的实际执行效果。
- E24 使用 IEEE-69 直接训练的冻结 EPS 模型，且采用 IEEE-123 单相结构等值，定位为规模外推审计；结果范围不包含 IEEE-123 直接训练和完整多相潮流验证。

---

""" + SPATIAL_OVERVIEW + "\n\n---\n\n" + e23 + "\n\n---\n\n" + e24
    text = _clean_document_language(text)
    (OUTPUT / "E23_E24_COMBINED.md").write_text(text, encoding="utf-8")


def _section_text(markdown: str) -> list[tuple[str, str, str | None]]:
    sections = re.split(r"(?m)^## ", markdown)
    parsed = []
    for section in sections[1:]:
        lines = section.splitlines()
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        image_match = re.search(r"!\[[^]]*\]\(figures/([^)]*?)\.png\)", body)
        image_name = image_match.group(1) if image_match else None
        body = re.sub(r"!\[[^]]*\]\([^)]*\)\n?", "", body)
        parsed.append((title, body, image_name))
    return parsed


def _add_markdown_body(document: Document, body: str) -> None:
    lines = body.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line or line == "---":
            index += 1
            continue
        if line.startswith("### "):
            document.add_heading(line[4:], level=3)
        elif line.startswith("- "):
            document.add_paragraph(line[2:], style="List Bullet")
        elif line.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            if len(table_lines) >= 2:
                headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
                data_lines = [row for row in table_lines[2:] if "---" not in row]
                table = document.add_table(rows=1, cols=len(headers))
                table.style = "Table Grid"
                for cell, value in zip(table.rows[0].cells, headers):
                    cell.text = value
                for data_line in data_lines:
                    values = [cell.strip() for cell in data_line.strip("|").split("|")]
                    cells = table.add_row().cells
                    for cell, value in zip(cells, values):
                        cell.text = value
            continue
        elif line.startswith("文件：") or line.startswith("文件:"):
            document.add_paragraph(line)
        elif not line.startswith("!["):
            document.add_paragraph(line)
        index += 1


def _build_docx() -> None:
    document = Document()
    document.sections[0].top_margin = Inches(0.6)
    document.sections[0].bottom_margin = Inches(0.6)
    document.sections[0].left_margin = Inches(0.7)
    document.sections[0].right_margin = Inches(0.7)
    title = document.add_heading("E23-E24：IEEE 配电网空间与规模约束实验汇总", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph("本报告合并 E23 的 IEEE-69 空间压力实验和 E24 的 IEEE-123 规模外推审计。")
    document.add_heading("E23 空间实验条件总览", level=1)
    _add_markdown_body(document, SPATIAL_OVERVIEW.replace("## E23 空间实验条件总览", "").strip())
    document.add_heading("实验关系", level=1)
    document.add_paragraph("E23 研究连续压力和空间异质性下的相对性能边界。连续压力部分使用 IEEE-69 直接训练的 EPS 模型，空间异质性部分使用 E21 aggregate EPS 模型；两部分均在 IEEE-69 上评估。E24 复用 IEEE-69 直接训练的 EPS 模型，并使用 IEEE-123 单相结构等值，因此不能解释为 IEEE-123 直接训练或完整多相潮流验证。")

    for label, source_text, figures in (("E23", _clean_document_language(_e23_description_source()), E23_FIGURES), ("E24", _clean_document_language(_e24_description_source()), E24_FIGURES)):
        document.add_page_break()
        document.add_heading(f"{label} 实验结果", level=0)
        source_sections = _section_text(source_text)
        source_by_image = {image: (title, body) for title, body, image in source_sections if image}
        for index, (name, title, short) in enumerate(figures, start=1):
            document.add_heading(f"{index}. {title}", level=1)
            image_path = (E23 if label == "E23" else E24) / "figures" / f"{name}.png"
            document.add_picture(str(image_path), width=Inches(6.7))
            document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            document.add_paragraph(short)
            if name in source_by_image:
                _add_markdown_body(document, source_by_image[name][1])
    document.save(OUTPUT / "E23_E24_COMBINED.docx")


def _add_slide(presentation: Presentation, image_path: Path, title: str, note: str, index: int, total: int) -> None:
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, PptInches(0), PptInches(0), PptInches(13.333), PptInches(0.7))
    header.fill.solid()
    header.fill.fore_color.rgb = RGBColor(31, 50, 72)
    header.line.fill.background()
    title_box = slide.shapes.add_textbox(PptInches(0.45), PptInches(0.12), PptInches(12.2), PptInches(0.4))
    paragraph = title_box.text_frame.paragraphs[0]
    paragraph.text = title
    paragraph.font.size = Pt(20)
    paragraph.font.color.rgb = RGBColor(255, 255, 255)
    with Image.open(image_path) as image:
        ratio = image.width / image.height
    area_width, area_height = 12.4, 5.75
    if ratio >= area_width / area_height:
        width, height = area_width, area_width / ratio
        left, top = 0.45, 0.88 + (area_height - height) / 2
    else:
        height, width = area_height, area_height * ratio
        left, top = 0.45 + (area_width - width) / 2, 0.88
    slide.shapes.add_picture(str(image_path), PptInches(left), PptInches(top), width=PptInches(width), height=PptInches(height))
    footer = slide.shapes.add_textbox(PptInches(0.55), PptInches(6.72), PptInches(12.0), PptInches(0.5))
    footer_paragraph = footer.text_frame.paragraphs[0]
    footer_paragraph.text = f"{note}｜第 {index}/{total} 页"
    footer_paragraph.font.size = Pt(10)
    footer_paragraph.font.color.rgb = RGBColor(90, 90, 90)


def _build_pptx() -> None:
    presentation = Presentation()
    presentation.slide_width = PptInches(13.333)
    presentation.slide_height = PptInches(7.5)
    presentation.core_properties.title = "E23-E24 IEEE 配电网空间与规模约束实验汇总"
    total = len(E23_FIGURES) + len(E24_FIGURES)
    index = 0
    for name, title, note in E23_FIGURES:
        index += 1
        _add_slide(presentation, E23 / "figures" / f"{name}.png", title, f"E23｜{note}", index, total)
    for name, title, note in E24_FIGURES:
        index += 1
        _add_slide(presentation, E24 / "figures" / f"{name}.png", title, f"E24｜{note}", index, total)
    presentation.save(OUTPUT / "E23_E24_COMBINED.pptx")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _build_markdown()
    _build_docx()
    _build_pptx()


if __name__ == "__main__":
    main()
