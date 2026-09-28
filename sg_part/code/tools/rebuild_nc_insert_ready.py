from pathlib import Path
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT=Path(__file__).resolve().parents[1]

def fmt(r,size=10,bold=False):
    r.font.name='Arial'; r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Arial'); r.font.size=Pt(size); r.bold=bold

def mr(text):
    r=OxmlElement('m:r'); t=OxmlElement('m:t'); t.text=text; r.append(t); return r

def math_token(token):
    # A compact native OMML builder for the subscript/superscript notation used here.
    if '_' in token or '^' in token:
        base=token.split('_')[0].split('^')[0]; sub=None; sup=None
        if '_' in token: sub=token.split('_',1)[1].split('^')[0].strip('{}')
        if '^' in token: sup=token.split('^',1)[1].strip('{}')
        if sub is not None and sup is not None:
            e=OxmlElement('m:sSubSup'); e.append(mr(base)); ss=OxmlElement('m:sub'); ss.append(mr(sub)); sp=OxmlElement('m:sup'); sp.append(mr(sup)); e.append(ss); e.append(sp); return e
        if sub is not None:
            e=OxmlElement('m:sSub'); e.append(mr(base)); ss=OxmlElement('m:sub'); ss.append(mr(sub)); e.append(ss); return e
        e=OxmlElement('m:sSup'); e.append(mr(base)); sp=OxmlElement('m:sup'); sp.append(mr(sup)); e.append(sp); return e
    return mr(token)

def add_math(p,expr):
    om=OxmlElement('m:oMath')
    for tok in expr.split(' '): om.append(math_token(tok))
    p._p.append(om)

def eq(d,expr):
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(2); p.paragraph_format.space_after=Pt(6); add_math(p,expr)

def paragraph(d,text,boldlead=None):
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after=Pt(6); p.paragraph_format.line_spacing=1.08
    if boldlead:
        r=p.add_run(boldlead); fmt(r,10,True)
    r=p.add_run(text); fmt(r,10)

def cell_text(c,text,bold=False,size=6.4):
    c.text=''; p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.0
    r=p.add_run(str(text)); fmt(r,size,bold); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    tc=c._tc; pr=tc.get_or_add_tcPr(); mar=OxmlElement('w:tcMar')
    for side,val in [('top','45'),('start','45'),('bottom','45'),('end','45')]:
        e=OxmlElement('w:'+side); e.set(qn('w:w'),val); e.set(qn('w:type'),'dxa'); mar.append(e)
    pr.append(mar); bd=OxmlElement('w:tcBorders')
    for edge in ('top','left','bottom','right','insideH','insideV'):
        e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'8'); e.set(qn('w:color'),'000000'); bd.append(e)
    pr.append(bd)

def cell_math(c,expr,size=6.2):
    c.text=''; p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(0); p.paragraph_format.line_spacing=1.0; add_math(p,expr); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    tc=c._tc; pr=tc.get_or_add_tcPr(); mar=OxmlElement('w:tcMar')
    for side,val in [('top','45'),('start','45'),('bottom','45'),('end','45')]:
        e=OxmlElement('w:'+side); e.set(qn('w:w'),val); e.set(qn('w:type'),'dxa'); mar.append(e)
    pr.append(mar); bd=OxmlElement('w:tcBorders')
    for edge in ('top','left','bottom','right','insideH','insideV'):
        e=OxmlElement('w:'+edge); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'8'); e.set(qn('w:color'),'000000'); bd.append(e)
    pr.append(bd)

def add_table(d,title,heads,rows,widths,size=6.3):
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(9); p.paragraph_format.space_after=Pt(5)
    r=p.add_run(title); fmt(r,11,True)
    t=d.add_table(rows=1,cols=len(heads)); t.autofit=False
    h=t.rows[0]; h.height=Inches(.48); h.height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
    rep=OxmlElement('w:tblHeader'); rep.set(qn('w:val'),'true'); h._tr.get_or_add_trPr().append(rep)
    for i,x in enumerate(heads): cell_text(h.cells[i],x,True,size); h.cells[i].width=Inches(widths[i])
    for row in rows:
        cells=t.add_row().cells; t.rows[-1].height=Inches(.35); t.rows[-1].height_rule=WD_ROW_HEIGHT_RULE.AT_LEAST
        for i,x in enumerate(row):
            if isinstance(x,tuple) and x[0]=='M': cell_math(cells[i],x[1],size)
            else: cell_text(cells[i],x,False,size)
            cells[i].width=Inches(widths[i])
    d.add_paragraph()

DATA=[
('P00','NextGen','device-day battery','5 min','p_i,t; q_i,t; s_i,t; E_i; P_i^max; a_i,t; z_i; t','100 sources; 3,000 device-days','battery and feeder reference'),
('P00b','data2','charging session','5 min','p_i,t; E_i; P_i^max; a_i,t; z_i; t','session/device proxy; g_i,t counterfactual','bidirectional charging stress'),
('P01','BDG1','building electricity','1 h','p_i,t; b_i,t; z_i; t','507 buildings; no measured g_i,t or s_i,t','building heterogeneity'),
('P02','Low Carbon London','household electricity','30 min','p_i,t; b_i,t; z_i; tariff_i; t','5,000 households; storage state simulated','large residential scale'),
('P03','BDG2','building electricity + solar','1 h','p_i,t; g_i,t; b_i,t; z_i; t','1,570 buildings; aligned g_i,t subset','measured/counterfactual mixture'),
('P04','Danish heat meters','thermal demand','1 h','h_i,t; b_i,t; z_i; t','2,400 meters; h_i,t is thermal','thermal response diversity'),
('P05','Smart Grid Smart City','load + generation','30 min','p_i,t; g_i,t; b_i,t; z_i; t','600 customers; general/controlled/gross/net channels','measured self-consumption'),
('P06','HEAPO','heat-pump electricity','15 min / daily','p_i,t; b_i,t; w_t; z_i; t','1,362 households; weather w_t','heat-pump heterogeneity'),
('P07','GoiEner','smart-meter electricity','1 h','p_i,t; b_i,t; c_i; z_i; t','5,000 users; c_i metadata only','large-user diversity'),
('P08','European LV urban-8k','LV active-power profile','24/168 points','p_i,t; q_i,t; n_i; z_i; t','5,000 customers; feeder/GIS context','urban feeder diversity'),
('P09','European LV rural','LV active/reactive profile','24/168 points','p_i,t; q_i,t; n_i; z_i; t','2,731 customers; rural topology','rural spatial diversity'),
('P10','European LV urban-35k','LV active/reactive profile','24/168 points','p_i,t; q_i,t; n_i; z_i; t','12,000 customers; urban topology','large urban network'),
('P11','Norway AMI','MV/LV AMI profile','1 h','p_i,t; q_i,t; g_i,t; n_i; w_t; z_i; t','2,994 sources; measured import/export','network-linked measured data'),
('P12','CAMSL-JP','smart-meter electricity','30 min','p_i,t; b_i,t; tariff_i; z_i; t','1,423 households; tariff groups','tariff/temporal response'),
('P13','Irish CER','import/export household','30 min','p_i,t; g_i,t; b_i,t; z_i; t','2,904 consumers; import/export','bidirectional profiles'),
('P14','OPSD','household multi-channel','15 min','p_i,t; g_i,t; b_i,t; s_i,t; z_i; t','11 households; source-limited','variable-rich small source'),
('P15','COMPLETE-EC','community load/PV/BESS/EV','15 min','p_i,t; g_i,t; b_i,t; s_i,t; E_i; z_i; t','250 members; PV/BESS/EV','integrated prosumer test')]

DATA_TEXT_EN={
'NextGen':'NextGen is the calibration population for the battery and feeder implementation. Its records contain device-day power trajectories, state variables, power limits and spatial assignments, which makes it the most complete source for checking the simulation chain. The independent-source inventory is approximately 100 devices, while the 3,000 device-day records repeat source identities over different days. The source is therefore used to validate parameter transfer, snapshot construction and network mechanisms; it is not used to claim 3,000 independent field devices.',
'data2':'The data2 adapter represents charging sessions at five-minute resolution. A session has a start and end boundary, so the availability indicator changes with the session window rather than being assigned as an unconditional scalar. The data are used to test bidirectional charging response and session-level participation. The external-input channel is counterfactual in this implementation, and the experiment keeps that provenance separate from measured generation.',
'BDG1':'BDG1 contains hourly electricity profiles for 507 buildings. The primary variation is the shape and timing of building demand, represented by the baseline trajectory and its time index. The source does not measure storage state, storage capacity or photovoltaic generation. When storage dispatch is simulated, those quantities are assigned by the explicit battery model, while the measured building trajectory remains the empirical driver.',
'Low Carbon London':'Low Carbon London supplies half-hourly household electricity and tariff metadata for a large residential population. The repeated daily structure permits comparisons of temporal variability, while household identity preserves cross-resource heterogeneity. The source does not provide a measured storage state for the simulated fleet. The storage variables are consequently model-side quantities and are interpreted separately from the observed household demand.',
'BDG2':'BDG2 extends building electricity with solar channels. A measured generation trajectory is used only where electricity and solar records are aligned; the remaining building records retain their measured load but use the explicitly labelled counterfactual input path. This split is important for distinguishing a result supported by observed generation from a result supported by load-shape diversity alone.',
'Danish heat meters':'The Danish source measures thermal demand rather than electrical power. It is included because the population-scale question concerns the reproducibility of aggregate flexible demand, and thermal demand supplies a distinct response family. The measured variable is kept as a thermal quantity in the analysis. An electrical interpretation requires an explicit heat-pump or COP transformation and is not silently assumed.',
'Smart Grid Smart City':'Smart Grid Smart City contains general supply, controlled load, gross generation and net generation channels for customers. These fields allow the measured demand and generation components to be separated before constructing an aggregate response. Its source inventory is smaller than the largest meter populations, so it contributes strong provenance and self-consumption evidence rather than only a large-N comparison.',
'HEAPO':'HEAPO provides household heat-pump electricity at 15-minute and daily resolutions, together with weather covariates. The weather series provides a common forcing variable that can be retained when analysing temporal dependence across households. The observed electrical trajectory remains distinct from the simulated storage state and capacity. The dataset therefore tests heat-pump demand heterogeneity under an explicit measurement boundary.',
'GoiEner':'GoiEner provides hourly smart-meter demand for a large user inventory and includes contract-power metadata. The self-generation category is treated as metadata and is not converted into a generation trajectory. This makes the dataset useful for load diversity and power-rating stratification, while preventing a categorical flag from being mistaken for measured photovoltaic output.',
'European LV urban-8k':'The urban-8k European LV source contributes active-power profiles together with feeder and GIS context. Its role is to test whether customer-profile diversity persists after resources are assigned to a physical network location. Reactive power is retained where available for the network audit, while the primary empirical response remains the active-power trajectory.',
'European LV rural':'The rural European LV source supplies paired active and reactive profiles on a rural topology. The joint profile is relevant because a change in active power can affect branch loading and voltage together with reactive power. The source therefore supports both population diversity analysis and spatially resolved network evaluation.',
'European LV urban-35k':'The urban-35k source provides the largest independent customer inventory in the table and includes active/reactive profiles with urban network context. Its contribution is not limited to the size of the source count: the P/Q structure and spatial information permit a direct comparison between statistical reproducibility and physical acceptance in a large urban representation.',
'Norway AMI':'Norway AMI combines measured active import/export, reactive power, weather and an anonymized MV/LV network model. It is the clearest source for connecting customer trajectories to a network representation while retaining measured-only provenance. The dataset supports analyses in which active and reactive quantities are evaluated together under feeder constraints.',
'CAMSL-JP':'CAMSL-JP contains half-hourly household consumption with tariff-group and temperature context. The tariff grouping supplies an observed categorical factor for temporal-response comparisons. There is no measured external-generation trajectory in the adapter, so the dataset is interpreted as a demand source and not as a photovoltaic validation source.',
'Irish CER':'Irish CER records active import and active export separately at 30-minute resolution. The separation preserves response direction and supports bidirectional household-profile tests. The anonymized files contain coverage gaps and outage periods; these are treated as properties of the source inventory rather than being used to create artificial independent sources.',
'OPSD':'OPSD contains a rich set of household channels, including load, photovoltaic, storage, electric-vehicle and heat-pump fields. At the same time, it contains only 11 independent households. This combination makes OPSD useful for demonstrating that a broad variable vector does not by itself provide a large empirical replication base.',
'COMPLETE-EC':'COMPLETE-EC describes an integrated energy community with load, photovoltaic, battery and electric-vehicle fields and explicit capacity or state variables for many assets. It is used to examine multi-technology coupling and community-level spatial assignments. Its 250-member inventory is interpreted as an integrated community case rather than as a large independent-source population.'}
ALG_ZH=[('无协调。','实现不发送协调信号，通信复杂度为 O(0)，设备沿用未协调基线。'),('本地 SOC 规则。','每设备读取 s、a 和 P^max 执行本地阈值，复杂度为 O(1)/device。'),('MPC。','使用 H=12 步预测和馈线剩余容量预算滚动优化，复杂度为 O(HN)。'),('均值场控制。','维护 K=10 个 SOC 分箱并每 15 分钟更新参与概率，复杂度为 O(N)。'),('虚拟电池。','维护等效能量和功率包络并按功率权重映射，复杂度为 O(N)。'),('分组能量管理。','设备提交不可拆分的 15 分钟 packet，复杂度为 O(N log N)。'),('交易式控制。','设备提交 bid，每 15 分钟离散价格清算，复杂度为 O(N log N)。'),('EPS 广播。','离线 Dual Quantile 估计器，日常发送单一广播，在线复杂度为 O(1)。'),('集中式贪心上限。','逐设备扫描 headroom 并线性分配，复杂度为 O(N)，仅作理论上限。')]

def build(lang):
    d=Document(); s=d.sections[0]; s.orientation=WD_ORIENT.LANDSCAPE; s.page_width,s.page_height=s.page_height,s.page_width; s.top_margin=s.bottom_margin=Inches(.5); s.left_margin=s.right_margin=Inches(.48)
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run('Supplementary Tables S1, S4 and S5' if lang=='en' else '补充表 S1、S4 和 S5'); fmt(r,16,True)
    if lang=='en':
        paragraph(d,'The empirical, control and network descriptions below are written as insert-ready supplementary results and methods text. The same notation is used across all three tables so that measured trajectories, simulated storage states and network variables remain distinguishable.')
        rows=[[a,b,c,e,f,g,h] for a,b,c,e,f,g,h in DATA]
        add_table(d,'Supplementary Table S1. Empirical population characteristics, standardized variables and data features.',['ID','Dataset','Resource / signal','Resolution','Variables','Provenance / limit','Data feature','N_src'],rows,[.32,.88,1.0,.55,2.0,1.45,1.35,.5],5.25)
        paragraph(d,'Table S1 establishes the empirical basis of the population-scale analysis. The source records differ in physical domain, measurement direction, temporal resolution, spatial context and the availability of generation or storage channels. These differences determine which part of the control chain is empirically constrained and which part is represented by the explicit storage model. In particular, measured demand, measured generation, thermal demand and simulated storage state are kept as separate evidence classes. This separation allows a result to be attributed to the observed source characteristic that supports it, rather than to the number of rows created after resampling.')
        paragraph(d,'The notation is shared by every source. A resource or charging session is indexed by i, a dispatch interval by t, and the sampling interval by τ. Active and reactive power are represented by p and q, state of charge by s, usable energy by E, maximum active power by P, availability by a, feeder or zone assignment by z, network context by n, external generation or charging input by g, baseline demand by b, weather forcing by w, contract power by c, and thermal demand by h. The independent-source count is retained separately from the number of device-day records.')
        eq(d,'P_t = Σ_i p_{i,t}   Q_t = Σ_i q_{i,t}   N_{src} = | { source_i } |')
        for name in [x[1] for x in DATA]: paragraph(d,DATA_TEXT_EN[name],boldlead=name+'. ')
        paragraph(d,'The pairwise mixed-population analysis uses 105 constructed populations formed from two empirical sources with equal weights of 0.5 and 0.5. A further 14 multi-source populations combine between 3 and 15 empirical sources with randomized composition weights. Each of the 15 empirical source populations contributes to 20–24 mixtures. These 119 populations are composition stress tests: they assess whether the observed response persists after source distributions are recombined, while the 15 published populations remain the primary empirical replication level. This distinction is preserved when interpreting confidence, between-population variation and source provenance.')
        urls=[['NextGen','project archive and local source documentation','battery and feeder reference'],['data2','project archive and local source documentation','charging-session source'],['BDG1','https://github.com/buds-lab/the-building-data-genome-project','Miller and Meggers (2017), DOI 10.1016/j.egypro.2017.07.400'],['Low Carbon London','https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households','London Datastore / UK Power Networks'],['BDG2','https://github.com/buds-lab/building-data-genome-project-2','Miller et al. (2020), DOI 10.1038/s41597-020-00712-x'],['Danish heat meters','https://zenodo.org/records/6563114','Schaffer et al. (2022), DOI 10.1038/s41597-022-01502-3'],['Smart Grid Smart City','https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef','Australian SGSC project'],['HEAPO','https://zenodo.org/records/15056919','arXiv:2503.16993'],['GoiEner','https://zenodo.org/records/7362094','WHY project / GoiEner'],['European LV urban-8k','https://data.mendeley.com/datasets/685vgp64sm/1','DOI 10.17632/685vgp64sm.1'],['European LV rural / urban-35k','https://data.mendeley.com/datasets/gspyzvvrhm/2','DOI 10.17632/gspyzvvrhm.2'],['Norway AMI','https://data.mendeley.com/datasets/jv3rz8k35r/1','Maree et al., AMI dataset'],['CAMSL-JP','https://data.mendeley.com/datasets/cmpsyncmmk/1','Kiguchi et al., CAMSL dataset'],['Irish CER','https://figshare.com/articles/dataset/6_150_years_1_month_17_days_and_8_hours_of_anonymised_electricity_smart_meter_data/31851922','DOI 10.6084/m9.figshare.31851922'],['OPSD','https://open-power-system-data.org/data-sources','Open Power System Data'],['COMPLETE-EC','https://zenodo.org/records/7602546','Faia et al. (2023), DOI 10.1016/j.dib.2023.109218']]
        add_table(d,'Data-source index used by Table S1.',['Dataset','Public source','Citation / use'],urls,[1.55,5.3,3.35],6.0)
        # S4
        add_table(d,'Supplementary Table S4. Implemented control strategies, required inputs and code-defined complexity.',['Strategy','Required input symbols','Online signal','Complexity','Implemented limitation'],[['No coordination','none','none','O(0)','unmanaged baseline'],['Local SOC rules','s_i,t, a_i,t, P_i^max','local state','O(1)/device','threshold and heterogeneity'],['MPC','r_t, s_i,t, E_i, P_i^max, q_i,t, network state','forecast + state','O(HN)','rolling optimization'],['Mean-field control','s_i,t, a_i,t, K=10 SOC bins','bin distribution','O(N)','correlation/closure'],['Virtual battery','s_i,t, E_i, P_i^max','equivalent envelope','O(N)','envelope clipping'],['Packetized Energy Management','s_i,t, a_i,t, 15-min packet','packet request','O(N log N)','packet granularity'],['Transactive control','s_i,t, load variation, bid','15-min price','O(N log N)','bid and clearing dependence'],['EPS broadcast','calibrated response estimator, aggregate feedback','single broadcast','O(1)','calibration and drift'],['Centralized greedy UB','device headroom, no communication budget','complete state','O(N)','theoretical reference only']],[1.35,2.35,1.4,1.0,3.5],5.8)
        paragraph(d,'Table S4 describes the implemented information contracts. The dispatch request is a scalar target and each strategy transforms that target using a different subset of local states, population summaries or network information. The equations below are the formal expressions used to describe those transformations. They are written as editable Word equations so that the notation can be retained when the text is moved into the manuscript.')
        eq(d,'p_{i,t} = u_t')
        eq(d,'p_{i,t} = clip ( u_t ; s_{i,t} , a_{i,t} , P_i^{max} )')
        eq(d,'min Σ_{t=1}^{H} ( P_t - r_t )^2   subject to   g ( x_{i,t} ) ≤ 0')
        eq(d,'m_t = ( 1 / N ) Σ_i φ ( x_{i,t} )   p_{i,t} = π ( x_{i,t} , m_t )')
        eq(d,'P_t = clip ( r_t , P_t^{min} ( m_t ) , P_t^{max} ( m_t ) )')
        eq(d,'A_{i,t} ~ Bernoulli ( ρ_t | s_{i,t} , a_{i,t} )')
        eq(d,'p_{i,t} = argmax_p [ v_i ( p ; x_{i,t} ) - λ_t p ]   Σ_i p_{i,t} = r_t')
        eq(d,'p_{i,t} = π_i ( b_t , x_{i,t} )   b_t = f ( r_t , P_t^{agg} )')
        eq(d,'max Σ_i v_i ( p_i )   subject to   g ( x ) ≤ 0')
        eq(d,'τ = 300 s   H = 12   K = 10   R^2_{min} = 0.95   p_{ctrl,min} = 0.90   N_{eff}^* = 500')
        paragraph(d,'No coordination provides the reference trajectory because no operator command is transmitted. Its code-defined communication complexity is O(0), and any response variability is attributed to the unmanaged baseline. The local SOC rule is evaluated independently at each resource. It reads the local state of charge, availability and power limit, applies the configured time-of-day and SOC thresholds, and produces a local action without reading another resource. The implementation is therefore O(1) per device, while the population contains N such local evaluations.')
        paragraph(d,'MPC uses the future availability and surplus prediction over H=12 intervals and applies the feeder-side remaining-capacity budget in the rolling water-level filling routine. The experiment code records O(HN) complexity. Its advantage is explicit use of the predicted trajectory and physical budget; its information requirement is correspondingly richer because the controller must maintain the forecast, device states and the network-side capacity state at each rolling step.')
        paragraph(d,'Mean-field control maintains K=10 SOC bins and, every 15 minutes, updates a common participation probability from the predicted bin distribution and expected availability. Devices then respond locally according to their current SOC bin. The recorded complexity is O(N), reflecting the population aggregation and the subsequent local actions. The approximation is informative when the bin state summarizes the fleet, while common forcing, strong dependence or drift can make the summary insufficient.')
        paragraph(d,'The virtual-battery implementation maintains an equivalent energy and power envelope and maps that envelope back to devices using fixed power weights. Its recorded complexity is O(N). This representation gives the operator a compact fleet-level state, but the actual device headroom remains heterogeneous; clipping therefore remains dataset and state dependent when the envelope does not match the current fleet.')
        paragraph(d,'Packetized Energy Management assigns an indivisible 15-minute fixed-power packet to a device. The aggregator accepts complete packets and an accepted packet remains active until its expiry. The experiment definition records O(N log N). The packet duration and the non-interruptible commitment create a clear operational interpretation, while packet timing and asynchronous expiry can produce tracking deviations when the requested trajectory changes faster than the packet schedule.')
        paragraph(d,'Transactive control constructs a bid from state of charge and load variation, clears the market every 15 minutes with a discrete price and keeps the resulting price while devices respond independently. Its recorded complexity is O(N log N), consistent with bid ordering and clearing. The response depends on the bid function, price quantization and the stability of the clearing sequence, so the comparison is interpreted as a market-style coordination baseline rather than as a universal optimizer.')
        paragraph(d,'EPS broadcast uses the offline Dual Quantile neural-network response estimator and sends a single broadcast command during routine operation. The experiment records O(1) online coordination complexity with respect to N. Local packet loss, offline status and Bernoulli response remain part of the device implementation, while aggregate feedback is used by the tested adaptation and audit procedures. The O(1) statement concerns routine operator-side communication; commissioning, calibration, drift detection and network screening remain separate activities.')
        paragraph(d,'The centralized greedy upper bound scans each device headroom and performs one linear greedy allocation, with code-defined complexity O(N). It assumes complete device information and does not impose a communication budget; the reported upper-bound calculation also treats network clipping as an audit layer rather than part of the device-headroom ceiling. It is therefore a theoretical information-unconstrained reference used to interpret the attainable effect, not a real implementation or a claim of universal optimality.')
        paragraph(d,'The numerical settings used by the benchmark are τ=300 s, H=12, K=10, R²_min=0.95, p_ctrl,min=0.90 and N_eff^*=500. Here τ is the simulation interval, H is the MPC prediction horizon, K is the number of SOC bins in the mean-field model, R²_min is the aggregate-response criterion, p_ctrl,min is the reliable-control criterion and N_eff^* is the effective-population reference used in the population-scale analysis.')
        # S5
        add_table(d,'Supplementary Table S5. Physical variables, network-scale constraints and validation environments.',['Environment','Scale','Physical variables','Mathematical limits','Primary validation role'],[['IEEE-33','33 buses; 32 branches','P_l,Q_l,V_k,S_l,T,z_i','L_l≤1; V_min≤V_k≤V_max; T≤1','identify active bottlenecks'],['IEEE-69','69 buses; 68 branches','P_l,Q_l,V_k,S_l,T,z_i,c_f,c_n','c_f=0.5; c_n=0.8; derating; M0-M6','topology and placement transfer'],['IEEE-123','123 buses; 122 branches','P_l,Q_l,V_k,S_l,T,z_i,A_k','voltage/transformer stress; audit limits','larger-scale feasibility']],[1.2,1.35,2.4,2.35,2.4],6.0)
        paragraph(d,'The three network environments share one physical realization layer. Each resource response is assigned to a bus using its spatial label, and the feeder calculation then evaluates active and reactive branch flow, bus voltage, transformer loading and the accepted portion of the aggregate request. This common procedure makes the comparison a test of physical deliverability after statistical coordination. It also preserves the distinction between a response that is accurately reproduced at the aggregate interface and a response that can be admitted by the feeder.')
        eq(d,'S_l = ( P_l^2 + Q_l^2 )^{1/2}   L_l = S_l / S_l^{max}   T = T_f / T^{max}')
        eq(d,'V_{min} ≤ V_k ≤ V_{max}   A_{net} = P_{acc} / P_{req}   C_{loss} = 1 - A_{net}')
        paragraph(d,'IEEE-33 is the smallest radial environment and is used to localize the first active physical bottleneck. The branch-flow variables P_l and Q_l determine the apparent flow S_l, while V_k identifies voltage sensitivity at each bus and T identifies transformer loading. The model supports nominal and stressed operating modes, so a response can be traced from the requested aggregate action to the particular inequality that clips it. The environment is therefore suited to mechanism-level interpretation of line, voltage and transformer effects.')
        paragraph(d,'IEEE-69 increases the radial path and provides a transfer environment for spatial placement. The feeder-concentration condition assigns c_f=0.5 of the resources to one feeder, while the distal-node condition assigns c_n=0.8 to end nodes. These distributions preserve the same aggregate request but change the shared upstream flow and the distance to voltage-sensitive nodes. The M0-M6 scenarios and the derating cases test whether a method that is statistically reliable retains its benefit after the response is embedded in a more spatially stressed topology.')
        paragraph(d,'IEEE-123 is the largest radial representation used in the audit. It retains the same variables and feasibility inequalities while increasing the number of buses and branches, allowing the analysis to examine whether the information-limited control architecture transfers to a larger network representation. The direct-training and transformer/voltage stress cases are interpreted as feasibility checks under the tested model, with bus-level acceptance used to identify where the requested response is reduced.')
        paragraph(d,'The physical validation changes one of three factors: a network parameter, a spatial distribution or the network scale. Parameter sweeps vary line capacity, transformer capacity or voltage bounds while the other protocol values remain fixed. Distribution sweeps compare uniform placement with feeder concentration and distal-node concentration. Scale sweeps compare IEEE-33, IEEE-69 and IEEE-123 under the common realization layer. The resulting comparison asks whether aggregate response fidelity remains high while A_net decreases, and which physical variable explains that separation. The environments therefore test the transfer from statistical coordination to deliverable power rather than treating topology as an additional empirical population.')
    else:
        # Generate the same structure with translated prose while retaining exactly the same six OMML equations.
        paragraph(d,'以下内容按照可直接插入 NC 补充正文的正式结构编写。三张表使用统一符号，使实测轨迹、仿真储能状态和网络物理变量保持清晰区分。')
        rows=[[a,b,c,e,f,g,h] for a,b,c,e,f,g,h in DATA]
        add_table(d,'补充表 S1。经验数据集特征、统一变量与数据特色。',['编号','数据集','资源/信号','分辨率','变量','来源/限制','数据特色','N_src'],rows,[.32,.88,1.0,.55,2.0,1.45,1.35,.5],5.25)
        paragraph(d,'表 S1 建立群体规模分析的经验数据基础。数据集在物理领域、计量方向、时间分辨率、空间信息以及发电和储能通道方面存在差异。这些差异决定了控制链中哪些部分受到实测数据约束，哪些部分由显式储能模型表示。实测负荷、实测发电、热需求和仿真储能状态被作为不同证据类别处理，因此跨数据集结果可以归因于具体观测特征，而不是简单归因于重采样后的行数。')
        paragraph(d,'全文统一使用同一套符号：i 表示资源或充电会话，t 表示调度时刻，τ 表示采样间隔；p 和 q 表示有功、无功功率，s 表示荷电状态，E 表示可用容量，P 表示最大有功功率，a 表示可用性，z 表示馈线或区域位置，n 表示网络/GIS 信息，g 表示外部发电或充电输入，b 表示基线需求，w 表示天气变量，c 表示合同功率，h 表示热需求。独立来源数与 device-day 记录数分开统计。')
        eq(d,'P_t = Σ_i p_{i,t}   Q_t = Σ_i q_{i,t}   N_{src} = | { source_i } |')
        for x in DATA:
            paragraph(d,{'NextGen':'NextGen 用于电池和馈线实现的校准，包含设备日功率轨迹、状态、功率限制和空间位置。约 100 个独立来源对应 3,000 个 device-day，后者不能解释为 3,000 个独立现场设备。','data2':'data2 以 5 分钟分辨率提供充电会话，起止时间形成明确可用性边界，用于双向充电响应和会话参与压力测试。当前适配器中的外部输入是反事实通道。','BDG1':'BDG1 提供 507 个建筑的小时用电，主要体现负荷形状和时间差异，不包含实测电池状态、容量和光伏发电；储能变量由显式模型赋值。','Low Carbon London':'Low Carbon London 提供 5,000 个家庭的半小时用电和费率信息，体现大规模住宅日内结构；储能状态属于仿真变量。','BDG2':'BDG2 在建筑用电之外提供太阳能通道。只有时间对齐的建筑使用实测 g，其余负荷记录使用明确标记的反事实输入。','Danish heat meters':'Danish 数据观测热需求 h，而不是电功率。它用于热需求群体规律测试；若需电气解释，必须另外声明热泵或 COP 转换。','Smart Grid Smart City':'SGSC 含一般负荷、受控负荷、总发电和净发电通道，可分离负荷与发电后构造聚合响应，主要支持实测自消费分析。','HEAPO':'HEAPO 提供 15 分钟和日尺度热泵用电及天气变量 w。电池状态和容量仍是仿真侧变量。','GoiEner':'GoiEner 提供大规模小时智能电表负荷和合同功率 c。self-generation 类别是元数据，不转换为发电轨迹。','European LV urban-8k':'European LV urban-8k 同时提供有功轮廓和馈线/GIS 信息，用于检验客户差异经过空间分配后的表现。','European LV rural':'European LV rural 提供农村拓扑上的有功和无功轮廓，可同时分析客户差异与电压、支路压力。','European LV urban-35k':'European LV urban-35k 的价值不仅是最大来源数，还在于大规模城市 P/Q 轮廓和网络信息能够检验可预测性与物理接受率的分离。','Norway AMI':'Norway AMI 结合实测输入/输出、有功/无功、天气和中低压网络模型，是网络关联的实测数据源。','CAMSL-JP':'CAMSL-JP 提供半小时用电、费率组和温度信息，用于费率暴露下的时间响应分析，不假定实测外部发电。','Irish CER':'Irish CER 分开记录输入和输出电量，使响应方向明确，适合双向家庭轮廓测试；缺失和停电作为来源条件保留。','OPSD':'OPSD 的负荷、光伏、储能、电动车和热泵变量较丰富，但只有 11 个独立家庭，体现变量丰富与来源不足的差异。','COMPLETE-EC':'COMPLETE-EC 含社区负荷、光伏、电池和电动车及容量/SOC 字段，适合多技术耦合；250 个成员属于综合社区案例。'}[x[1]],boldlead=x[1]+'. ')
        paragraph(d,'pairwise mixed 分析包含 105 个双源混合群体，每个来源权重为 0.5；另外 14 个多源混合群体组合 3–15 个经验来源并使用随机组成权重。15 个经验来源分别出现在 20–24 个混合群体中。这 119 个群体用于检验来源重组后的稳健性，15 个发表数据集仍是主要经验复制层。')
        paragraph(d,'表 S4 描述各实现的信息契约。代码中的复杂度分别为 O(0)、O(1)/device、O(HN)、O(N)、O(N)、O(N log N)、O(N log N)、O(1) 和 O(N)，对应代码中无协调、本地 SOC、MPC、均值场、虚拟电池、packetized、交易式、EPS 和集中式贪心上限。')
        add_table(d,'补充表 S4。已实现控制策略、所需输入与代码定义的复杂度。',['策略','所需输入符号','在线信号','复杂度','实现局限'],[['无协调','无','无','O(0)','未协调基线'],['本地 SOC 规则','s_i,t,a_i,t,P_i^max','本地状态','O(1)/device','阈值与异质性'],['MPC','r_t,s_i,t,E_i,P_i^max,q_i,t,网络状态','预测+状态','O(HN)','滚动优化'],['均值场控制','s_i,t,a_i,t,K=10 分箱','分箱分布','O(N)','相关/闭合'],['虚拟电池','s_i,t,E_i,P_i^max','等效包络','O(N)','包络截断'],['分组能量管理','s_i,t,a_i,t,15-min packet','packet 请求','O(N log N)','packet 粒度'],['交易式控制','s_i,t,负荷波动,bid','15-min 价格','O(N log N)','竞价/清算依赖'],['EPS 广播','离线校准估计器,聚合反馈','单一广播','O(1)','标定/漂移'],['集中式贪心 UB','设备 headroom,无通信预算','完整状态','O(N)','理论参考']], [1.35,2.35,1.4,1.0,3.5],5.8)
        paragraph(d,'控制请求是标量目标，算法根据本地状态、群体摘要或网络信息完成转换。以下公式与英文版保持完全相同，均以 Word 原生公式对象插入。')
        for e in ['p_{i,t} = u_t','p_{i,t} = clip ( u_t ; s_{i,t} , a_{i,t} , P_i^{max} )','min Σ_{t=1}^{H} ( P_t - r_t )^2   subject to   g ( x_{i,t} ) ≤ 0','m_t = ( 1 / N ) Σ_i φ ( x_{i,t} )   p_{i,t} = π ( x_{i,t} , m_t )','P_t = clip ( r_t , P_t^{min} ( m_t ) , P_t^{max} ( m_t ) )','A_{i,t} ~ Bernoulli ( ρ_t | s_{i,t} , a_{i,t} )','p_{i,t} = argmax_p [ v_i ( p ; x_{i,t} ) - λ_t p ]   Σ_i p_{i,t} = r_t','p_{i,t} = π_i ( b_t , x_{i,t} )   b_t = f ( r_t , P_t^{agg} )','max Σ_i v_i ( p_i )   subject to   g ( x ) ≤ 0','τ = 300 s   H = 12   K = 10   R^2_{min} = 0.95   p_{ctrl,min} = 0.90   N_{eff}^* = 500']: eq(d,e)
        for lead,txt in ALG_ZH: paragraph(d,txt,boldlead=lead)
        paragraph(d,'集中式贪心上限读取全部设备 headroom，不设置通信预算，代码中复杂度为 O(N)。它用于表示信息不受限时的理论设备侧吸收上限，不是可部署控制器，也不是普适最优性的声明。')
        add_table(d,'补充表 S5。物理变量、网络规模约束与验证环境。',['环境','规模','物理变量','数学限制','主要验证作用'],[['IEEE-33','33 母线；32 支路','P_l,Q_l,V_k,S_l,T,z_i','L_l≤1; V_min≤V_k≤V_max; T≤1','识别活跃瓶颈'],['IEEE-69','69 母线；68 支路','P_l,Q_l,V_k,S_l,T,z_i,c_f,c_n','c_f=0.5; c_n=0.8; 降额; M0-M6','拓扑与布置迁移'],['IEEE-123','123 母线；122 支路','P_l,Q_l,V_k,S_l,T,z_i,A_k','电压/变压器压力；审计限制','较大规模可行性']], [1.2,1.35,2.4,2.35,2.4],6.0)
        paragraph(d,'三个网络环境共享同一物理实现层。设备响应通过 z_i 映射到母线，馈线计算随后评估有功/无功支路潮流、母线电压、变压器负载和被接受的聚合请求。该共同流程用于检验统计协调之后的物理可交付性，并区分聚合接口上的响应保真度与馈线能够接纳的响应。')
        for e in ['S_l = ( P_l^2 + Q_l^2 )^{1/2}   L_l = S_l / S_l^{max}   T = T_f / T^{max}','V_{min} ≤ V_k ≤ V_{max}   A_{net} = P_{acc} / P_{req}   C_{loss} = 1 - A_{net}']: eq(d,e)
        paragraph(d,'IEEE-33 是最小的径向网络，用于定位首先活跃的物理瓶颈。P_l 和 Q_l 决定视在潮流 S_l，V_k 表示母线电压敏感性，T 表示变压器负载。正常和压力运行模式使请求响应可以沿着“聚合请求—潮流—约束截断”链路追踪，适合解释线路、电压和变压器作用。')
        paragraph(d,'IEEE-69 增加径向路径并用于空间布置迁移。馈线集中条件令 c_f=0.5 的资源位于一条馈线，末端节点条件令 c_n=0.8 的资源位于末端母线。相同的聚合请求因共享上游潮流和末端电压敏感性而产生不同的物理接受率；M0-M6 与降额条件检验统计可靠性在空间压力下是否仍保留可交付效果。')
        paragraph(d,'IEEE-123 是审计中最大的径向网络表示，保留相同变量和可行性不等式，同时增加母线与支路数量。直接训练以及变压器/电压压力条件用于检验低信息控制架构向更大网络表示的迁移，母线级接受率用于定位请求响应被削减的位置。')
        paragraph(d,'物理验证改变三类因素：网络参数、空间分布和环境规模。参数扫描逐项改变线路容量、变压器容量或电压边界；分布扫描比较均匀布置、馈线集中和末端节点集中；规模扫描比较 IEEE-33、IEEE-69 与 IEEE-123 的共同实现层。由此可以检验 R² 保持较高时 A_net 是否下降，以及哪一个物理变量解释二者的分离。')
    d.save(ROOT/('outputs/Supplementary_Tables_S1_S4_S5_English.docx' if lang=='en' else 'outputs/Supplementary_Tables_S1_S4_S5_Chinese.docx'))

if __name__=='__main__': build('en'); build('zh')
