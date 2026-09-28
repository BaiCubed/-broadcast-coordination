from __future__ import annotations

from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figs" / "supplementary_s1_s4_s5"
OUT.mkdir(parents=True, exist_ok=True)
DOCX_PATH = ROOT / "outputs" / "Supplementary_Tables_S1_S4_S5_English.docx"

NAVY = "17324D"
BLUE = "2F6B9A"
TEAL = "2A9D8F"
ORANGE = "E38B4A"
GREY = "697586"
LIGHT = "EAF0F5"
RED = "B84C4C"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, **kwargs) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in kwargs:
            tag = "w:" + edge
            element = borders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                borders.append(element)
            for key in ["val", "sz", "space", "color"]:
                if key in kwargs[edge]:
                    element.set(qn("w:" + key), str(kwargs[edge][key]))


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths=None, font_size=7.3):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.autofit = False
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for i, text in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = text
        set_cell_shading(cell, NAVY)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(font_size)
    for ridx, values in enumerate(rows):
        cells = table.add_row().cells
        for i, text in enumerate(values):
            cells[i].text = str(text)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ridx % 2 == 0:
                set_cell_shading(cells[i], "F5F8FA")
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(font_size)
        for c in cells:
            set_cell_border(c, top={"val": "single", "sz": 3, "color": "C9D2DC"},
                            bottom={"val": "single", "sz": 3, "color": "C9D2DC"},
                            left={"val": "single", "sz": 3, "color": "C9D2DC"},
                            right={"val": "single", "sz": 3, "color": "C9D2DC"})
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph()
    return table


def add_heading(doc: Document, text: str, level=1):
    p = doc.add_paragraph()
    p.style = f"Heading {level}"
    p.paragraph_format.space_before = Pt(8 if level == 1 else 5)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.color.rgb = RGBColor.from_string(NAVY)
    return p


def add_body(doc: Document, text: str, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.08
    r = p.add_run(text)
    r.font.size = Pt(9.2)
    r.italic = italic
    return p


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(8.3)
    r.font.color.rgb = RGBColor.from_string(GREY)


def figure_s1(path: Path) -> None:
    labels = ["BDG1", "LCL", "BDG2", "DK heat", "SGSC", "HEAPO", "GoiEner",
              "EU-LV urban-8k", "EU-LV rural", "EU-LV urban-35k", "Norway AMI",
              "CAMSL-JP", "Irish CER", "OPSD", "COMPLETE-EC"]
    counts = [507, 5000, 1570, 2400, 600, 1362, 5000, 5000, 2731, 12000, 2994, 1423, 2904, 11, 250]
    cats = ["building", "meter", "building", "thermal", "meter", "thermal", "meter", "feeder", "feeder", "feeder", "feeder", "meter", "meter", "meter", "building"]
    colors = {"building": "#" + BLUE, "meter": "#" + TEAL, "thermal": "#" + ORANGE, "feeder": "#7B61A8"}
    order = np.argsort(counts)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    y = np.arange(len(labels))
    ax.barh(y, np.array(counts)[order], color=[colors[cats[i]] for i in order], edgecolor="white")
    ax.set_yticks(y, np.array(labels)[order], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Available source count (log scale)")
    ax.axvspan(2, 3000, color="#DCEAF4", alpha=0.8, label="tested population sizes: N = 2-3,000")
    ax.axvline(3000, color="#" + RED, lw=1.1, ls="--")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Empirical source inventory", fontsize=10, weight="bold")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    from matplotlib.patches import Patch
    handles = [Patch(color=v, label=k.title()) for k, v in colors.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, ncol=2)
    ax2 = axes[1]
    ax2.bar(["Pairwise", "Multi-source"], [105, 14], color=["#" + BLUE, "#" + ORANGE], width=0.6)
    ax2.set_ylabel("Number of constructed populations")
    ax2.set_ylim(0, 125)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_title("Mixed-population construction", fontsize=10, weight="bold")
    for x, v in enumerate([105, 14]):
        ax2.text(x, v + 3, str(v), ha="center", fontsize=10, weight="bold")
    ax2.text(0, 78, "two sources; weight 0.5 + 0.5", ha="center", fontsize=8)
    ax2.text(1, 28, "3-15 sources; randomized weights", ha="center", fontsize=8)
    ax2.text(0.5, -0.18, "Each empirical source appears in 20-24 mixtures", transform=ax2.transAxes,
             ha="center", fontsize=8, color="#" + GREY)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_s4(path: Path) -> None:
    f = ROOT / "outputs" / "figs" / "subpanel" / "source_data" / "06_ieee69_all_algorithms_m0_m6.csv"
    d = pd.read_csv(f)
    d = d[d["algorithm"] != "centralized_optimal"].copy()
    order = ["no_coordination", "local_rules", "mpc_optimal", "mean_field_control", "virtual_battery",
             "packetized_energy_management", "transactive_control", "eps_ieee69_fused"]
    names = {"no_coordination": "Open-loop / no coordination", "local_rules": "Local SOC rule", "mpc_optimal": "MPC",
             "mean_field_control": "Mean-field control", "virtual_battery": "Virtual battery",
             "packetized_energy_management": "Packetized energy", "transactive_control": "Transactive control",
             "eps_ieee69_fused": "EPS broadcast (hybrid)"}
    med = d.groupby("algorithm")[["response_r2", "network_acceptance_ratio"]].median().reindex(order)
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    x = np.arange(len(order))
    ax.bar(x - 0.19, med["response_r2"], 0.36, label="Aggregate response R2", color="#" + BLUE)
    ax.bar(x + 0.19, med["network_acceptance_ratio"], 0.36, label="Network acceptance", color="#" + TEAL)
    ax.axhline(0.95, color="#" + ORANGE, ls="--", lw=1, label="R2 reference = 0.95")
    ax.set_xticks(x, [names[k] for k in order], rotation=28, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Median ratio across dataset-condition cells")
    ax.set_title("Information assumptions are distinct from outcome metrics", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_s5(path: Path) -> None:
    f = ROOT / "outputs" / "figs" / "subpanel" / "source_data" / "03_ieee69_spatial_retention.csv"
    d = pd.read_csv(f)
    d = d[d["algorithm"] == "eps_ieee69_fused"].copy()
    med = d.groupby(["topology", "stress_mode"])["retention"].median().unstack()
    topo = ["ieee33", "ieee69", "ieee123"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    ax.bar(x - 0.19, med.reindex(topo)["M5"], 0.38, label="50% feeder concentration (M5)", color="#" + BLUE)
    ax.bar(x + 0.19, med.reindex(topo)["M6"], 0.38, label="80% distal-node concentration (M6)", color="#" + RED)
    ax.axhline(0.8, color="#" + ORANGE, ls="--", lw=1, label="80% retention reference")
    ax.set_xticks(x, ["IEEE-33\n33 nodes / 32 branches", "IEEE-69\n69 nodes / 68 branches", "IEEE-123\n123 nodes / 122 branches"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Retention relative to uniform placement")
    ax.set_title("Physical deliverability depends on topology and placement", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def setup_doc() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width
    section.top_margin = Inches(0.55)
    section.bottom_margin = Inches(0.55)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)
    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(9.2)
    for style_name in ["Heading 1", "Heading 2"]:
        styles[style_name].font.name = "Arial"
        styles[style_name].font.bold = True
    styles["Heading 1"].font.size = Pt(14)
    styles["Heading 2"].font.size = Pt(11)
    return doc


def main() -> None:
    s1 = OUT / "Figure_S1_source_inventory_and_mixtures.png"
    s4 = OUT / "Figure_S4_information_assumptions.png"
    s5 = OUT / "Figure_S5_network_realization.png"
    figure_s1(s1)
    figure_s4(s4)
    figure_s5(s5)

    doc = setup_doc()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Supplementary Tables S1, S4 and S5")
    r.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = RGBColor.from_string(NAVY)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = sub.add_run("Population-scale coordination: empirical basis, information assumptions and network realization")
    rr.font.size = Pt(10)
    rr.font.color.rgb = RGBColor.from_string(GREY)
    add_body(doc, "This document consolidates the three supplementary tables identified in the table-design notes. Each table is paired with a focused figure and a short interpretation. Constructed mixed populations are treated as composition stress tests and are not counted as additional independent empirical datasets.")

    add_heading(doc, "Supplementary Table S1 | Empirical populations and mixed-population construction", 1)
    add_body(doc, "The purpose of Table S1 is to show that the population-scale sweep is anchored in heterogeneous empirical sources rather than in one favourable dataset. The source-count column reports the inventory used by the local preprocessing audit; it is distinct from the number of simulated device-days. The role column identifies the evidentiary use of each source, while the final index column provides a stable place for the paper or URL to be inserted during manuscript finalization.")
    s1_rows = [
        ["P01", "BDG1", "Building", "Measured building electricity", "507", "N-scale anchor; smallest building inventory", "S1-01"],
        ["P02", "Low Carbon London", "Meter", "Measured household electricity", "5,000", "Large household-load population", "S1-02"],
        ["P03", "BDG2", "Building", "Measured electricity; partial measured solar", "1,570", "Mixed measured/counterfactual sensitivity", "S1-03"],
        ["P04", "Danish heat meters", "Thermal", "Measured district-heating demand", "2,400", "Thermal-demand proxy; not direct electric load", "S1-04"],
        ["P05", "Smart Grid Smart City", "Meter", "Measured load and generation fields", "600", "Measured-input and self-consumption evidence", "S1-05"],
        ["P06", "HEAPO", "Thermal", "Measured household heat-pump electricity", "1,362", "Heat-pump demand heterogeneity", "S1-06"],
        ["P07", "GoiEner", "Meter", "Measured smart-meter electricity", "5,000", "Large-user population; load diversity", "S1-07"],
        ["P08", "European LV urban-8k", "Feeder", "Measured P profiles with LV context", "5,000", "Urban feeder customer profiles", "S1-08"],
        ["P09", "European LV rural", "Feeder", "Measured P/Q profiles and topology", "2,731", "Rural network customer profiles", "S1-09"],
        ["P10", "European LV urban-35k", "Feeder", "Measured P/Q profiles and topology", "12,000", "Large urban feeder population", "S1-10"],
        ["P11", "Norway AMI", "Feeder", "Measured AMI active/reactive power", "2,994", "Measured-only network-linked profiles", "S1-11"],
        ["P12", "CAMSL-JP", "Meter", "Measured half-hourly smart-meter load", "1,423", "Tariff-response and temporal diversity", "S1-12"],
        ["P13", "Irish CER", "Meter", "Measured import/export smart-meter load", "2,904", "Bidirectional household profiles", "S1-13"],
        ["P14", "OPSD", "Meter", "Measured household channels", "11", "Small-source stress test; explicitly limited", "S1-14"],
        ["P15", "COMPLETE-EC", "Building", "Measured community load, PV, BESS and EV fields", "250", "Integrated community stress test", "S1-15"],
    ]
    add_table(doc, ["ID", "Population", "Resource category", "Observed data", "Available source count", "Role in the study", "Reference index"], s1_rows,
              widths=[0.42, 1.25, 0.9, 2.2, 0.9, 2.6, 0.8], font_size=6.9)
    doc.add_picture(str(s1), width=Inches(10.0))
    add_caption(doc, "Figure S1 | Empirical source inventory and mixed-population construction. The shaded interval marks the physical population sizes used in the scale sweep (N = 2-3,000). Source counts refer to distinct source identifiers available to the preprocessing audit; simulated device-days are not substituted for independent sources.")
    add_heading(doc, "Mixed-population construction summary", 2)
    add_table(doc, ["Mixture class", "Number", "Source populations", "Construction rule", "Interpretation"], [
        ["Pairwise mixture", "105", "Two empirical sources", "Equal weights (0.5 + 0.5)", "Composition robustness to reweighting two sources"],
        ["Multi-source mixture", "14", "3-15 empirical sources", "Randomized source weights", "Broader heterogeneity and dependence stress test"],
        ["Reuse balance", "119 total", "15 published sources", "Each source appears in 20-24 mixtures", "Prevents one source from dominating the ensemble"],
    ], widths=[1.35, 0.7, 1.8, 2.2, 3.3], font_size=7.2)
    add_body(doc, "Figure S1 makes two checks visible. First, the tested scale interval is embedded in the empirical inventory for most sources, while the explicitly small OPSD and COMPLETE-EC populations remain visible as limited-source stress tests. Second, the mixed-population ensemble is balanced by construction: 105 pairwise mixtures provide a controlled two-source comparison and 14 multi-source mixtures broaden the composition space. Results based on these 119 mixtures therefore support robustness to source composition, whereas the 15 published populations provide the empirical-diversity evidence.")

    add_heading(doc, "Supplementary Table S4 | Control strategies and information assumptions", 1)
    add_body(doc, "Table S4 is organized around information access rather than an algorithm-performance ranking. The comparison distinguishes what is available during routine operation from what is required for calibration, forecasting or network screening. The centralized greedy controller is retained as a full-information reference; it is not interpreted as a universal optimum across all network and device constraints.")
    s4_rows = [
        ["Open-loop baseline", "Predefined signal", "No feedback during dispatch", "None online", "Lower reference for response variability", "No adaptation to state or network"],
        ["Local SOC rule", "Device-local heuristic", "Device SOC and local limits", "Device state only", "Device-level control baseline", "No fleet-wide state required"],
        ["MPC", "Optimization", "Predictions, constraints and state estimates", "Forecasts plus device/network model", "Optimization reference", "Information-intensive and horizon-dependent"],
        ["Hybrid / EPS broadcast", "Broadcast plus local response", "Broadcast signal and aggregate measurements", "Offline calibration; aggregate online adaptation", "Practical reduced-observability strategy", "Routine coordination dimension is fleet-size independent"],
        ["Centralized greedy reference", "Full-information dispatch", "All device states and network constraints", "Complete fleet and feeder state", "Upper reference for deliverability", "Reference implementation, not a universal bound"],
    ]
    add_table(doc, ["Strategy", "Control type", "Information required for action", "Calibration / planning information", "Purpose", "Limitation to state explicitly"], s4_rows,
              widths=[1.35, 1.25, 2.2, 2.0, 2.0, 2.25], font_size=7.0)
    doc.add_picture(str(s4), width=Inches(10.0))
    add_caption(doc, "Figure S4 | Median aggregate-response fidelity and network acceptance across dataset-condition cells in the IEEE network audit. The plot is descriptive: it visualizes the information-assumption comparison and does not define a universal algorithm ranking.")
    add_body(doc, "The information boundary is the main result of Table S4. Open-loop and local rules require little or no operator-side information but provide limited adaptation. MPC and centralized references use richer state and constraint information. The hybrid/EPS row separates routine online broadcast coordination from offline registration and model calibration, so the phrase 'without real-time device-level sensing' is not read as 'without any device information at any stage'. The outcome plot is included to show why information access and physical performance should be reported as separate dimensions.")

    add_heading(doc, "Supplementary Table S5 | Physical realization and network evaluation settings", 1)
    add_body(doc, "Table S5 maps the population-level response onto radial distribution-network representations. The table reports the topology scale and the evaluation purpose, while the figure emphasizes the physical effect that matters for interpretation: a statistically controllable aggregate response can lose deliverability when resources are concentrated or feeder headroom is reduced.")
    s5_rows = [
        ["Reference feeder", "Reference case", "Reference representation", "Aggregate threshold robustness", "No topology-specific performance claim"],
        ["IEEE-33", "33", "32", "Network validation and branch/voltage screening", "Radial feeder with explicit line, voltage and transformer limits"],
        ["IEEE-69", "69", "68", "Transfer and spatial-placement evaluation", "Larger radial feeder for topology and concentration stress"],
        ["IEEE-123", "123", "122", "Safety audit and direct-training evaluation", "Highest-resolution tested feeder representation"],
    ]
    add_table(doc, ["Network", "Nodes", "Branches", "Evaluation purpose", "Interpretation boundary"], s5_rows,
              widths=[1.5, 0.85, 0.9, 3.0, 4.0], font_size=7.2)
    add_table(doc, ["Scenario family", "Spatial condition", "Primary quantity", "Meaning"], [
        ["Nominal / matched", "Uniform placement", "Network acceptance and aggregate-response fidelity", "Separates statistical response quality from normal feeder headroom"],
        ["Feeder concentration", "50% of resources concentrated on one feeder (M5)", "Retention relative to uniform placement", "Tests shared upstream branch and transformer bottlenecks"],
        ["Distal-node concentration", "80% of resources placed at distal nodes (M6)", "Retention relative to uniform placement", "Tests voltage sensitivity and end-of-feeder stress"],
        ["Constraint sweep", "Line, transformer and voltage limits varied", "Acceptance, curtailment and response fidelity", "Identifies which physical limits modify deliverability"],
    ], widths=[1.6, 2.7, 2.9, 3.05], font_size=7.2)
    doc.add_picture(str(s5), width=Inches(8.7))
    add_caption(doc, "Figure S5 | EPS retention under spatial concentration on three feeder representations. Retention is normalized to uniform placement; values below one indicate loss of physically deliverable benefit after network screening.")
    add_body(doc, "The network evaluation is a feasibility layer, not a replacement for the aggregate-control analysis. The topology comparison supports reuse of a common effective-population coordinate for the aggregate threshold within the tested feeder set, while the spatial-placement results show that physical acceptance and retained curtailment benefit remain topology- and placement-dependent. Accordingly, Table S5 should be read as evidence for constrained simulation feasibility under the listed IEEE models, not as field-safety validation or as a universal ranking of control methods.")

    add_heading(doc, "Reference index for Table S1 (to be completed in the manuscript)", 1)
    add_body(doc, "The index below is intentionally separated from the main table so that citation formatting can be changed without disturbing the data summary. Verified public URLs are included where they are already recorded in the project documentation; entries marked 'add paper/URL' are placeholders for the final bibliography.")
    refs = [
        ["S1-01", "BDG1", "https://github.com/buds-lab/the-building-data-genome-project", "Miller and Meggers, Energy Procedia (2017), DOI 10.1016/j.egypro.2017.07.400"],
        ["S1-02", "Low Carbon London", "https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households", "Add final project citation"],
        ["S1-03", "BDG2", "https://github.com/buds-lab/building-data-genome-project-2", "Miller et al., Scientific Data (2020), DOI 10.1038/s41597-020-00712-x"],
        ["S1-04", "Danish heat meters", "https://zenodo.org/records/6563114", "Schaffer et al., Scientific Data (2022), DOI 10.1038/s41597-022-01502-3"],
        ["S1-05", "Smart Grid Smart City", "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef", "Add final project citation"],
        ["S1-06", "HEAPO", "https://zenodo.org/records/15056919", "HEAPO preprint, arXiv:2503.16993"],
        ["S1-07", "GoiEner", "https://zenodo.org/records/7362094", "Add final project citation"],
        ["S1-08", "European LV urban-8k", "https://data.mendeley.com/datasets/685vgp64sm/1", "DOI 10.17632/685vgp64sm.1"],
        ["S1-09", "European LV rural", "https://data.mendeley.com/datasets/gspyzvvrhm/2", "DOI 10.17632/gspyzvvrhm.2"],
        ["S1-10", "European LV urban-35k", "https://data.mendeley.com/datasets/gspyzvvrhm/2", "DOI 10.17632/gspyzvvrhm.2"],
        ["S1-11", "Norway AMI", "https://data.mendeley.com/datasets/jv3rz8k35r/1", "Maree, Energy distribution models with AMI smart meter sensor dataset"],
        ["S1-12", "CAMSL-JP", "https://data.mendeley.com/datasets/cmpsyncmmk/1", "Kiguchi et al., CAMSL dataset"],
        ["S1-13", "Irish CER", "https://figshare.com/articles/dataset/6_150_years_1_month_17_days_and_8_hours_of_anonymised_electricity_smart_meter_data/31851922", "Smeaton, Cai and Chen, DOI 10.6084/m9.figshare.31851922"],
        ["S1-14", "OPSD", "https://open-power-system-data.org/data-sources", "Open Power System Data household data; verify version"],
        ["S1-15", "COMPLETE-EC", "https://zenodo.org/records/7602546", "Faia et al., Data in Brief (2023), DOI 10.1016/j.dib.2023.109218"],
    ]
    add_table(doc, ["Index", "Dataset", "Paper or public URL", "Citation note"], refs,
              widths=[0.75, 1.65, 5.3, 3.5], font_size=6.9)
    add_body(doc, "Data provenance note: Danish heat-meter observations are reported as thermal-demand measurements, and load-only datasets with no measured external-generation channel are not presented as measured photovoltaic input. Small-source datasets are retained as explicit stress tests rather than silently pooled with large empirical populations.", italic=True)
    doc.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
