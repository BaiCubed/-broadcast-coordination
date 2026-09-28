"""生成E22全部图片的通俗版Markdown和Word说明。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[4]
E22 = ROOT / "results" / "E22"
MD_OUTPUT = E22 / "FIGURE_DESCRIPTIONS.md"
DOCX_OUTPUT = E22 / "FIGURE_DESCRIPTIONS.docx"
COMBINED_STRESS = E22 / "figures/ieee69_stress_profiles_comparison.png"
COMBINED_ALGORITHMS = E22 / "figures/trained_all_algorithms_comparison.png"
LEGACY_OVERALL = E22 / "trained_eps_ieee69/data/combined_e22_overall.csv"
DIRECT_OVERALL = E22 / "trained_eps_ieee69_direct/data/combined_e22_overall.csv"

MERGED_ALGORITHM_ORDER = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_e20_pooled",
    "eps_e21_mixed",
    "eps_in_domain",
    "eps_ieee69_formula_only",
    "eps_ieee69_learned_only",
    "eps_ieee69_fused",
    "centralized_optimal",
)

MERGED_ALGORITHM_COLORS = {
    "no_coordination": "#9c9c9c",
    "local_rules": "#ed7d31",
    "mpc_optimal": "#7057ff",
    "mean_field_control": "#4e79a7",
    "virtual_battery": "#59a14f",
    "packetized_energy_management": "#af7aa1",
    "transactive_control": "#d55e00",
    "eps_e20_pooled": "#ff9f40",
    "eps_e21_mixed": "#069c8f",
    "eps_in_domain": "#2a7fb8",
    "eps_ieee69_formula_only": "#767676",
    "eps_ieee69_learned_only": "#6f4aa8",
    "eps_ieee69_fused": "#1f5aa6",
    "centralized_optimal": "#3a9d66",
}

SCENARIOS = [
    ("M0", "常规运行", "按节点负荷分布设备；使用正常线路容量和较温和负荷，作为普通运行参照。"),
    ("M1", "深馈线高负载", "提高负荷并收紧线路容量，观察远端线路更容易过载时算法还能交付多少响应。"),
    ("M2", "双瓶颈反向潮流", "在两个远端区域设置能源富余，同时收紧线路容量，模拟富余能源经过多条受限支路传输。"),
    ("M3", "降额和预测误差", "进一步收紧线路容量，并让控制看到的预测与实际输入存在偏差，代表较困难的复合扰动。"),
    ("M4", "设备均匀分布", "恢复常规网络条件，将5000台设备尽量平均放到所有非平衡节点，作为空间部署参照。"),
    ("M5", "50%集中在远端馈线", "其他条件与M4相同，只把一半设备集中到预先固定的远端馈线，模拟社区级集中。"),
    ("M6", "80%集中在远端节点", "其他条件与M4相同，只把80%设备集中到一个远端节点，模拟充电站或工业园。"),
]

ALGORITHMS = [
    ("No coordination", "O(0)", "不发送控制命令，用作没有协调时的零效果参照。"),
    ("Local SOC rules", "O(1) per device", "每台设备只看自己的SOC、可用状态和时段规则，不知道全网富余。"),
    ("MPC", "O(HN)", "使用未来12步预测滚动安排设备，信息多、计算量也较高。"),
    ("Mean-field control", "O(N)", "把设备按SOC分组，用群体平均状态决定统一参与概率。"),
    ("Virtual battery", "O(N)", "把全部设备近似成一块大电池，再将聚合功率分回各设备。"),
    ("Packetized Energy Management", "O(N log N)", "设备申请固定时长的充电包，聚合器选择接受哪些完整包。"),
    ("Transactive control", "O(N log N)", "设备提交价格和功率意愿，聚合器通过市场清算决定参与。"),
    ("E20 pooled EPS", "O(1)", "将多个源数据的响应样本池化训练，测试时只广播一个控制信号。"),
    ("E21 mixed EPS", "O(1)", "直接使用多数据集混合场景训练，强调复杂场景下的统一广播。"),
    ("EPS in-domain", "O(1)", "只在目标数据集上训练，用来判断域内训练是否优于迁移模型。"),
    ("EPS formula-only", "O(1)", "直接训练实验中的消融版本，只使用解析公式，不使用学习模型。"),
    ("EPS learned-only", "O(1)", "只使用IEEE-69直接训练模型，不使用解析公式。"),
    ("EPS IEEE-69 fused", "O(1)", "融合解析公式、直接训练模型和闭环短缺修正。"),
    ("Centralized greedy UB", "O(N)", "逐设备统计可用充电空间，计算设备层最多能吸收多少富余；不经过网络裁剪，不能当作物理可执行结果。"),
]


def _main_entries() -> list[dict[str, Any]]:
    script = "src/extra/ieee33_device_day_simulation/figures/run_e22_ieee69_complexity.py"
    return [
        {
            "section": "一、Reviewer #4与配网约束主实验",
            "title": "IEEE-69约束分项审计",
            "image": "figures/ieee69_constraint_component_audit.png",
            "pdf": "figures/ieee69_constraint_component_audit.pdf",
            "data": ["data/e22_summary.csv"],
            "script": script,
            "purpose": "把审稿人关心的线路、电压、主变和安全层作用分开显示，避免只用一个总效果数字回答所有问题。每个格子汇总17个数据集和30个paired seeds。",
            "axes": [
                "横坐标：M0-M6。M0-M3改变网络压力，M4-M6改变设备空间位置，具体含义见文档前面的场景表。",
                "纵坐标：10种主实验算法，每一行代表一种不同控制方法。",
                "左上颜色和数字：执行后的最大支路负载率；1.0等于线路热限，大于1.0表示仍有过载。",
                "右上颜色和数字：执行后的最低节点电压；允许下限是0.95 p.u.，越低风险越大。",
                "左下颜色和数字：执行后的最大主变负载率；1.0等于6 MVA主变容量上限。",
                "右下颜色和数字：安全层使无违规时间比例提高了多少个百分点；数值大说明原始请求需要较多修正。",
            ],
            "reading": "该图直接回答Reviewer #4提出的三类物理约束是否被显式检查。结果不能简单概括为“全部安全”：部分场景仍可能出现支路超限或接近电压下限，正确结论是E22识别并量化了网络边界。",
            "limit": "仍使用平衡单相LinDistFlow和标准测试馈线，不等于三相AC潮流、保护校核或现场试验。",
        },
        {
            "section": "一、Reviewer #4与配网约束主实验",
            "title": "IEEE-69效果与原始请求安全性",
            "image": "figures/ieee69_effect_safety_pareto.png",
            "pdf": "figures/ieee69_effect_safety_pareto.pdf",
            "data": ["data/e22_overall_summary.csv"],
            "script": script,
            "purpose": "把“最终吸收了多少富余能源”和“算法原始请求是否会触发网络违规”放在同一张图中，防止把安全层修正后的安全结果误写成算法天然安全。",
            "axes": [
                "七个子图：分别对应M0-M6，每个子图使用相同坐标范围。",
                "横坐标：经过设备和网络约束后的弃电降低率，越右表示实际效果越高。Centralized greedy UB例外，它是未经过网络裁剪的设备层理论值。",
                "纵坐标：安全层介入前，算法原始请求不造成线路、电压或主变违规的时间比例；100%表示所有时间步的原始请求都无违规。",
                "散点颜色：代表算法，右下空白子图中的统一图例给出算法名称和复杂度。",
            ],
            "reading": "右上区域最好，表示效果高且原始请求本身较安全。靠右但较低的点表示效果请求很强，但依赖安全层大量裁剪；靠上但较左表示请求安全但能源利用效果有限。",
            "limit": "Greedy UB只作设备层上界参考，不能用它证明网络可交付性。",
        },
        {
            "section": "一、Reviewer #4与配网约束主实验",
            "title": "IEEE-69网络可接受响应比例",
            "image": "figures/ieee69_network_acceptance.png",
            "pdf": "figures/ieee69_network_acceptance.pdf",
            "data": ["data/e22_summary.csv"],
            "script": script,
            "purpose": "直接量化算法想让设备响应的功率中，有多少能够通过线路热限、电压范围和主变容量的联合校核。",
            "axes": [
                "横坐标：M0-M6场景。M1-M3是网络压力，M5-M6是空间集中。",
                "纵坐标：主实验中的10种算法。",
                "颜色和格内百分数：网络接受率，即网络最终接受的响应除以算法原始请求；100%表示全部请求可以执行。",
            ],
            "reading": "数值越低，说明聚合层命令与设备所在位置或网络剩余容量越不匹配。Greedy行表示其理论请求中实际能通过网络校核的比例，而不是将Greedy上界视为可交付结果。",
            "limit": "这是三类约束共同作用后的总接受率，具体是哪一类约束造成下降要结合约束分项审计图。",
        },
        {
            "section": "一、Reviewer #4与配网约束主实验",
            "title": "IEEE-69支路约束的时空位置",
            "image": "figures/ieee69_branch_time_heatmap.png",
            "pdf": "figures/ieee69_branch_time_heatmap.pdf",
            "data": ["data/ieee69_m2_eps_trace.json"],
            "script": script,
            "purpose": "用一个具体案例展示网络瓶颈发生在一天中的哪个时间、哪条支路，而不是只报告全天平均值。案例固定为CEC、M2、E21 mixed EPS和第一个seed。",
            "axes": [
                "横坐标：一天中的288个五分钟时间步；从左到右表示时间推进。",
                "纵坐标：IEEE-69的68条支路，按网络配置顺序排列。",
                "颜色：支路负载率；1.0等于线路热限，颜色越亮表示越接近或超过热限。",
            ],
            "reading": "连续亮带表示某些支路在一段时间内持续接近瓶颈；局部亮点表示短时拥塞。它证明统一广播的影响具有明确的空间位置和时间位置。",
            "limit": "这是单数据集、单场景、单seed案例，只回答线路热限的时空分布，不能单独证明电压和主变安全。",
        },
        {
            "section": "二、拓扑、压力和空间集中",
            "title": "IEEE-69全部算法的网络与空间压力响应",
            "image": "figures/ieee69_stress_profiles_comparison.png",
            "pdf": "figures/ieee69_stress_profiles_comparison.pdf",
            "data": [
                "trained_eps_ieee69/data/combined_e22_overall.csv",
                "trained_eps_ieee69_direct/data/combined_e22_overall.csv",
            ],
            "script": "src/extra/ieee33_device_day_simulation/figures/generate_e22_figure_descriptions.py",
            "purpose": "把原E22基线、迁移EPS、目标域内训练EPS和三种IEEE-69直接训练消融放在同一个坐标系中，直接比较它们面对网络压力和设备空间集中时的变化。",
            "axes": [
                "横坐标：M0-M6；M0是常规参照，M1-M3逐渐增加网络困难，M4-M6专门比较设备空间分布。",
                "纵坐标：17个数据集汇总后的弃电降低率，单位为百分比，越高表示吸收的富余能源越多。",
                "14条曲线：7种传统基线、E20/E21迁移EPS、目标域内训练EPS、3种IEEE-69直接训练消融和Greedy设备层上界；图例括号中给出在线复杂度。",
            ],
            "reading": "所有算法共享相同横纵坐标，可以直接比较同一场景下的高低。同一条线从M0到M3下降说明网络压力削弱效果；从M4到M6下降说明空间集中削弱效果。",
            "limit": "该图重点是效果，不显示具体违反的是线路、电压还是主变约束。",
        },
        {
            "section": "二、拓扑、压力和空间集中",
            "title": "不同数据集的拓扑效果保持率",
            "image": "figures/retention_by_dataset_algorithm.png",
            "pdf": "figures/retention_by_dataset_algorithm.pdf",
            "data": ["data/e22_retention.csv"],
            "script": script,
            "purpose": "逐数据集检查从IEEE-33换成IEEE-69后还保留多少效果，避免跨数据集平均值掩盖个别数据集的明显下降。",
            "axes": [
                "横坐标：10种主实验算法。",
                "纵坐标：17个数据集。",
                "颜色：保持率=IEEE-69效果/IEEE-33效果；1表示效果不变，小于1表示复杂网络下效果下降，大于1只表示两个网络的空间匹配不同。",
            ],
            "reading": "同一算法一整列颜色接近，说明拓扑影响较稳定；同一数据集一整行偏低，说明该数据的时序或空间负荷更容易受到IEEE-69瓶颈影响。",
            "limit": "保持率大于1不能简单解释为IEEE-69更容易；Greedy列也不用于支持网络安全。",
        },
        {
            "section": "二、拓扑、压力和空间集中",
            "title": "设备空间集中造成的效果损失",
            "image": "figures/spatial_concentration_retention.png",
            "pdf": "figures/spatial_concentration_retention.pdf",
            "data": ["data/e22_spatial_retention.csv"],
            "script": script,
            "purpose": "专门回答大量设备集中在同一社区馈线或同一节点时，统一广播效果是否仍能保持。M4-M6除了设备位置外全部成对固定。",
            "axes": [
                "横坐标：M5为50%设备集中在远端馈线，M6为80%设备集中在远端节点。",
                "纵坐标：10种主实验算法。",
                "左面板：IEEE-33；右面板：IEEE-69。",
                "颜色和数字：相对均匀部署M4的效果保持率；1表示不变，小于1表示集中部署造成损失。",
            ],
            "reading": "保持率明显低于1说明设备数量很大也不能消除局部电网瓶颈，空间位置必须进入控制可实施性判断。",
            "limit": "Greedy只表示设备层能力，不用于证明空间集中后网络仍然安全。",
        },
        {
            "section": "二、拓扑、压力和空间集中",
            "title": "IEEE-33与IEEE-69成对效果对比",
            "image": "figures/topology_effect_dumbbell.png",
            "pdf": "figures/topology_effect_dumbbell.pdf",
            "data": ["data/e22_overall_summary.csv"],
            "script": script,
            "purpose": "在相同设备、数据、场景和seed下，只改变馈线拓扑，直接观察更复杂的IEEE-69是否改变算法效果。",
            "axes": [
                "横坐标：跨17个数据集和M0-M6汇总的弃电降低率，单位为百分比。",
                "纵坐标：算法名称和在线复杂度。",
                "蓝点：IEEE-33；红点：IEEE-69；两点之间的线越长，表示拓扑变化造成的效果差异越大。",
            ],
            "reading": "红点明显位于蓝点左侧，表示更复杂拓扑降低可交付效果；两点接近表示该算法对拓扑变化较不敏感。",
            "limit": "Greedy点是设备层上界，不经过网络裁剪，因此只能作理论参照。",
        },
    ]


def _read_overall(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merged_algorithm_data() -> dict[str, dict[str, Any]]:
    legacy_rows = _read_overall(LEGACY_OVERALL)
    direct_rows = _read_overall(DIRECT_OVERALL)
    merged: dict[str, dict[str, Any]] = {}
    for algorithm in MERGED_ALGORITHM_ORDER:
        source = legacy_rows if algorithm == "eps_in_domain" else direct_rows
        rows = [
            row
            for row in source
            if row["topology"] == "ieee69" and row["algorithm"] == algorithm
        ]
        if not rows:
            raise ValueError(f"缺少IEEE-69算法结果：{algorithm}")
        values = {row["stress_mode"]: float(row["curtailment_reduction_pct"]) for row in rows}
        missing = [code for code, _, _ in SCENARIOS if code not in values]
        if missing:
            raise ValueError(f"{algorithm}缺少场景：{', '.join(missing)}")
        label = "EPS in-domain" if algorithm == "eps_in_domain" else rows[0]["algorithm_label"]
        merged[algorithm] = {
            "label": label,
            "complexity": rows[0]["complexity"],
            "values": values,
        }
    return merged


def _save_figure(fig: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_combined_stress(data: dict[str, dict[str, Any]]) -> None:
    scenario_codes = [code for code, _, _ in SCENARIOS]
    x = np.arange(len(scenario_codes))
    markers = ("o", "s", "^", "D", "v", "P", "X", "o", "s", "^", "D", "v", "P", "X")
    fig, axis = plt.subplots(figsize=(14.2, 7.2))
    for index, algorithm in enumerate(MERGED_ALGORITHM_ORDER):
        item = data[algorithm]
        axis.plot(
            x,
            [item["values"][code] for code in scenario_codes],
            color=MERGED_ALGORITHM_COLORS[algorithm],
            marker=markers[index],
            linewidth=2.0,
            markersize=5.5,
            label=f"{item['label']} ({item['complexity']})",
        )
    axis.set_xticks(x, scenario_codes)
    axis.set_ylim(0, 110)
    axis.set_ylabel("Deliverable curtailment reduction (%)")
    axis.set_xlabel("IEEE-69 network and spatial stress scenario")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, COMBINED_STRESS)


def _plot_combined_algorithms(data: dict[str, dict[str, Any]]) -> None:
    values = [float(np.mean(list(data[algorithm]["values"].values()))) for algorithm in MERGED_ALGORITHM_ORDER]
    y = np.arange(len(MERGED_ALGORITHM_ORDER))
    fig, axis = plt.subplots(figsize=(12.4, 7.8))
    axis.barh(
        y,
        values,
        color=[MERGED_ALGORITHM_COLORS[algorithm] for algorithm in MERGED_ALGORITHM_ORDER],
    )
    axis.set_yticks(
        y,
        [
            f"{data[algorithm]['label']}  {data[algorithm]['complexity']}"
            for algorithm in MERGED_ALGORITHM_ORDER
        ],
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 110)
    for index, value in enumerate(values):
        axis.text(value + 0.6, index, f"{value:.2f}%", va="center", fontsize=8)
    axis.set_xlabel("Mean deliverable curtailment reduction across M0-M6 (%)")
    axis.set_title("IEEE-69 comparison across baselines and all EPS training variants")
    axis.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    _save_figure(fig, COMBINED_ALGORITHMS)


def _compose_comparison_figures() -> None:
    data = _merged_algorithm_data()
    _plot_combined_stress(data)
    _plot_combined_algorithms(data)


def _r2_entries() -> list[dict[str, Any]]:
    data_path = E22 / "interim/r2_scaling/data/e22_completed_r2_points.csv"
    with data_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    script = "src/extra/ieee33_device_day_simulation/figures/plot_e22_interim_r2_scaling.py"
    return [{
        "section": "三、R²与设备规模阶段审计",
        "title": "五个已完成数据集的R²规模组图",
        "image": "interim/r2_scaling/figures/e22_completed_r2_group.png",
        "pdf": "interim/r2_scaling/figures/e22_completed_r2_group.pdf",
        "data": ["interim/r2_scaling/data/e22_completed_r2_points.csv"],
        "script": script,
        "purpose": "展示BDG1、BDG2、CEC、Danish和EU-Rural在设备数量增加时，独立测试R²如何提高。这里只包含已有完整Figure 3A规模结果的5个数据集，不代表全部17个数据集；单数据集子图不在本版说明中。",
        "axes": [
            "横坐标：物理设备数量N，使用对数刻度；向右表示聚合设备越来越多。",
            "纵坐标：独立测试R²；1表示预测与真实响应非常一致，0表示不优于使用均值，负值表示更差。",
            "蓝线和圆点：实际采样的R²；浅蓝阴影：95%置信区间。",
            "红色横虚线：R²=0.95目标；绿色竖虚线：沿log10(N)插值得到的N95。",
        ],
        "reading": "曲线越早穿过0.95，说明较少设备就能形成稳定聚合响应。该图回答统计规模问题，不直接回答线路、电压和主变安全问题。",
        "limit": "它复用Figure 3A独立测试协议，不是E22固定N=5000的response_r2重绘。",
    }]


def _trained_entries() -> list[dict[str, Any]]:
    script = "src/extra/ieee33_device_day_simulation/figures/run_e22_trained_eps_comparison.py"
    legacy_root = "trained_eps_ieee69"
    direct_root = "trained_eps_ieee69_direct"
    return [
        {
            "section": "四、迁移模型与IEEE-69直接训练",
            "title": "逐数据集比较迁移和IEEE-69直接训练",
            "image": f"{direct_root}/figures/trained_vs_transfer_dataset_heatmap_ieee69.png",
            "pdf": f"{direct_root}/figures/trained_vs_transfer_dataset_heatmap_ieee69.pdf",
            "data": [f"{direct_root}/data/combined_e22_summary.csv", f"{direct_root}/data/combined_e22_overall.csv"],
            "script": script,
            "purpose": "用一张父集热图比较两种迁移EPS和三种IEEE-69直接训练控制。原来的三列迁移/域内训练子集热图不再单独保留。",
            "axes": [
                "横坐标：E20 pooled、E21 mixed、formula-only、learned-only和fused五种EPS。",
                "纵坐标：17个数据集。",
                "颜色和数字：IEEE-69上M0-M6平均弃电降低率。",
            ],
            "reading": "同一行横向比较可看出该数据集更适合迁移、公式控制、学习控制还是融合控制；同一列纵向比较可看出同一种方法在不同数据集上的稳定性。",
            "limit": "所有方法仍在仿真网络中测试，直接训练不等于现场训练。",
        },
        {
            "section": "四、迁移模型与目标数据域内训练",
            "title": "域内训练相对E21迁移的逐数据集增益",
            "image": f"{legacy_root}/figures/trained_gain_by_dataset_ieee69.png",
            "pdf": f"{legacy_root}/figures/trained_gain_by_dataset_ieee69.pdf",
            "data": [f"{legacy_root}/data/combined_e22_summary.csv", f"{legacy_root}/data/combined_e22_overall.csv"],
            "script": script,
            "purpose": "直接显示在每个数据集上，目标数据域内训练比E21 mixed迁移模型多或少获得多少弃电降低率。",
            "axes": [
                "横坐标：效果差值，单位为百分点；正值表示域内训练更好，负值表示E21迁移更好。",
                "纵坐标：17个数据集，按增益从低到高排列。",
                "条形颜色：区分正增益和负增益。",
            ],
            "reading": "它帮助识别哪些数据集值得重新训练，哪些数据集直接迁移已经足够。",
            "limit": "训练增益不等于网络安全提升。",
        },
        {
            "section": "五、IEEE-69直接训练与控制消融",
            "title": "直接训练fused EPS相对E21迁移的增益",
            "image": f"{direct_root}/figures/trained_gain_by_dataset_ieee69.png",
            "pdf": f"{direct_root}/figures/trained_gain_by_dataset_ieee69.pdf",
            "data": [f"{direct_root}/data/combined_e22_summary.csv", f"{direct_root}/data/combined_e22_overall.csv"],
            "script": script,
            "purpose": "逐数据集比较IEEE-69直接训练fused EPS和E21 mixed迁移模型。",
            "axes": [
                "横坐标：直接训练fused EPS减去E21 mixed迁移效果，单位为百分点。",
                "纵坐标：17个数据集，按增益排序。",
                "正值：直接训练更好；负值：迁移模型更好。",
            ],
            "reading": "该图说明直接训练并不保证在所有数据集上改善，训练分布、数据量和模型误差都可能影响结果。",
            "limit": "这里只比较效果，不把差值解释为安全性差值。",
        },
        {
            "section": "五、IEEE-69直接训练与控制消融",
            "title": "IEEE-69全部算法平均效果对比",
            "image": "figures/trained_all_algorithms_comparison.png",
            "pdf": "figures/trained_all_algorithms_comparison.pdf",
            "data": [
                f"{legacy_root}/data/combined_e22_overall.csv",
                f"{direct_root}/data/combined_e22_overall.csv",
            ],
            "script": script,
            "purpose": "把所有传统基线、迁移EPS、目标域内训练EPS和IEEE-69直接训练消融放在同一张全算法图中。由于横纵坐标相同，不再使用左右分栏。",
            "axes": [
                "横坐标：IEEE-69上跨17个数据集和M0-M6平均弃电降低率。",
                "纵坐标：14种算法，包含传统基线、E20/E21迁移、EPS in-domain、formula-only、learned-only、fused和Greedy上界。",
                "条形末端数字：对应算法的平均效果；纵轴文字末尾给出复杂度。",
            ],
            "reading": "E20 pooled使用池化迁移模型；E21 mixed使用混合场景模型；EPS in-domain只在目标数据集训练；formula-only只用公式；learned-only只用直接训练模型；fused融合公式、模型和闭环修正。EPS版本均保持O(1)在线广播。",
            "limit": "Centralized greedy UB是不经过网络裁剪的O(N)设备层理论上界，不应与可交付算法作同口径安全解释。",
        },
    ]


def _entries() -> list[dict[str, Any]]:
    return [*_main_entries(), *_r2_entries(), *_trained_entries()]


def _overview_markdown() -> list[str]:
    lines = [
        "# E22 全部图片通俗说明",
        "",
        "本说明覆盖本版保留的E22汇总图片。主实验使用17个数据集、IEEE-33/IEEE-69、M0-M6、30个paired seeds和14种对比算法。文档重点区分三件事：算法想响应多少、网络允许执行多少、最终真正减少多少弃电。",
        "",
        "## 先理解四个常用指标",
        "",
        "- **弃电降低率**：原本会被浪费的富余能源中，有多少最终被设备吸收。100%表示全部吸收。",
        "- **网络接受率**：算法请求的响应中，有多少通过线路、电压和主变校核后能够执行。",
        "- **请求无违规比例**：不经过安全层时，算法原始请求有多少时间步不会造成网络违规。",
        "- **执行无违规比例**：经过安全层修正后，实际执行结果有多少时间步无违规。",
        "- **R²**：预测变化与真实变化的一致程度；越接近1越好，它不是简单的平方，因此可以为负。",
        "",
        "## M0-M6场景是什么意思",
        "",
        "|场景|通俗名称|具体含义|",
        "|-|-|-|",
    ]
    lines.extend(f"|{code}|{name}|{description}|" for code, name, description in SCENARIOS)
    lines.extend([
        "",
        "## 不同算法的简单介绍",
        "",
        "|算法|在线复杂度|通俗说明|",
        "|-|-|-|",
    ])
    lines.extend(f"|{name}|{complexity}|{description}|" for name, complexity, description in ALGORITHMS)
    lines.extend([
        "",
        "> 重要：`Centralized greedy UB` 是设备层理论上界，不经过网络事后裁剪。其他算法图中的“可交付效果”和Greedy上界不是完全相同口径。",
        "",
    ])
    return lines


def _write_markdown(entries: list[dict[str, Any]]) -> None:
    lines = _overview_markdown()
    current_section = None
    for index, entry in enumerate(entries, start=1):
        if entry["section"] != current_section:
            current_section = entry["section"]
            lines.extend([f"# {current_section}", ""])
        lines.extend([
            f"## {index}. {entry['title']}",
            "",
            f"![{entry['title']}]({entry['image']})",
            "",
            "文件：",
            "",
            f"- PNG：`{entry['image']}`",
            f"- PDF：`{entry['pdf']}`",
        ])
        lines.extend(f"- 数据：`{path}`" for path in entry["data"])
        lines.extend([
            f"- 生成脚本：`{entry['script']}`",
            "",
            "### 这张图想回答什么",
            "",
            entry["purpose"],
            "",
            "### 横纵坐标和图中元素",
            "",
        ])
        lines.extend(f"- {item}" for item in entry["axes"])
        lines.extend([
            "",
            "### 应该怎么读",
            "",
            entry["reading"],
            "",
            "### 不能说明什么",
            "",
            entry["limit"],
            "",
        ])
    MD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def _set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = text
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Microsoft YaHei"
            run.font.size = Pt(9)
            run.font.bold = bold
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def _add_bullet(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.add_run(text)


def _write_docx(entries: list[dict[str, Any]]) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)
    styles = document.styles
    styles["Normal"].font.name = "Microsoft YaHei"
    styles["Normal"].font.size = Pt(10)
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for style_name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        styles[style_name].font.name = "Microsoft YaHei"
        styles[style_name]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

    title = document.add_heading("E22 全部图片通俗说明", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph = document.add_paragraph(
        "本说明覆盖本版保留的E22汇总图片。主实验使用17个数据集、IEEE-33/IEEE-69、M0-M6、30个paired seeds和14种对比算法。"
    )
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    document.add_heading("先理解四个常用指标", level=1)
    for text in (
        "弃电降低率：原本会被浪费的富余能源中，有多少最终被设备吸收。",
        "网络接受率：算法请求的响应中，有多少通过线路、电压和主变校核后能够执行。",
        "请求无违规比例：安全层介入前，算法原始请求有多少时间步无网络违规。",
        "执行无违规比例：经过安全层修正后，实际执行有多少时间步无违规。",
        "R²：预测变化与真实变化的一致程度，越接近1越好，也可能为负。",
    ):
        _add_bullet(document, text)

    document.add_heading("M0-M6场景是什么意思", level=1)
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, ("场景", "通俗名称", "具体含义")):
        _set_cell_text(cell, text, True)
    for code, name, description in SCENARIOS:
        cells = table.add_row().cells
        for cell, text in zip(cells, (code, name, description)):
            _set_cell_text(cell, text)

    document.add_heading("不同算法的简单介绍", level=1)
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, ("算法", "在线复杂度", "通俗说明")):
        _set_cell_text(cell, text, True)
    for name, complexity, description in ALGORITHMS:
        cells = table.add_row().cells
        for cell, text in zip(cells, (name, complexity, description)):
            _set_cell_text(cell, text)
    warning = document.add_paragraph()
    run = warning.add_run(
        "重要：Centralized greedy UB是设备层理论上界，不经过网络事后裁剪，不能当作物理可执行结果。"
    )
    run.bold = True

    current_section = None
    for index, entry in enumerate(entries, start=1):
        document.add_page_break()
        if entry["section"] != current_section:
            current_section = entry["section"]
            document.add_heading(current_section, level=1)
        document.add_heading(f"{index}. {entry['title']}", level=2)
        image_path = E22 / entry["image"]
        if image_path.is_file():
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run()
            run.add_picture(str(image_path), width=Inches(6.5))
        document.add_heading("文件与来源", level=3)
        _add_bullet(document, f"PNG：{entry['image']}")
        _add_bullet(document, f"PDF：{entry['pdf']}")
        for path in entry["data"]:
            _add_bullet(document, f"数据：{path}")
        _add_bullet(document, f"生成脚本：{entry['script']}")
        document.add_heading("这张图想回答什么", level=3)
        document.add_paragraph(entry["purpose"])
        document.add_heading("横纵坐标和图中元素", level=3)
        for item in entry["axes"]:
            _add_bullet(document, item)
        document.add_heading("应该怎么读", level=3)
        document.add_paragraph(entry["reading"])
        document.add_heading("不能说明什么", level=3)
        document.add_paragraph(entry["limit"])

    core = document.core_properties
    core.title = "E22全部图片通俗说明"
    core.subject = "E22主实验、R²阶段图、迁移训练和直接训练图片说明"
    core.author = "Broadcast Coordination Project"
    document.save(DOCX_OUTPUT)


def generate() -> tuple[Path, Path, int]:
    _compose_comparison_figures()
    entries = _entries()
    missing = [entry["image"] for entry in entries if not (E22 / entry["image"]).is_file()]
    if missing:
        raise FileNotFoundError(f"以下图片不存在：{missing}")
    _write_markdown(entries)
    _write_docx(entries)
    return MD_OUTPUT, DOCX_OUTPUT, len(entries)


if __name__ == "__main__":
    markdown, word, count = generate()
    print(f"已生成Markdown：{markdown}")
    print(f"已生成Word：{word}")
    print(f"图片数量：{count}")
