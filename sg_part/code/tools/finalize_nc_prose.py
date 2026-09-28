from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT=Path(__file__).resolve().parents[1]

def runfmt(r,size=10,bold=False):
    r.font.name='Arial'; r._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Arial'); r.font.size=Pt(size); r.bold=bold
def omath(p,expr):
    o=OxmlElement('m:oMath')
    for tok in expr.split(' '):
        mr=OxmlElement('m:r'); mt=OxmlElement('m:t'); mt.text=tok; mr.append(mt); o.append(mr)
    p._p.append(o)
def para(d,text,boldlead=None):
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after=Pt(6); p.paragraph_format.line_spacing=1.08
    if boldlead:
        r=p.add_run(boldlead); runfmt(r,10,True)
    r=p.add_run(text); runfmt(r,10)
def eq(d,expr):
    p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(6); omath(p,expr)

ALG_EN=[
('No coordination. ','The implementation sends no coordination signal, so its operator-side communication complexity is O(0). Each resource follows its unmanaged baseline and provides the lower reference for measuring the effect of coordination.'),
('Local SOC rules. ','At each step, a device applies a local threshold rule using s_{i,t}, a_{i,t} and P_i^max. The implementation complexity is O(1) per device. It is transparent and scalable at the device level, but it cannot use fleet-wide response information and its result depends on threshold and availability heterogeneity.'),
('MPC. ','The rolling controller uses H=12 future intervals, a data-driven availability forecast and the feeder-side remaining-capacity budget. Its implementation complexity is O(HN), as recorded in the experiment definitions. MPC can enforce a richer set of device and network constraints, while its operation requires state forecasts and repeated optimization.'),
('Mean-field control. ','The implementation maintains K=10 SOC bins and updates a common participation probability every 15 minutes from the predicted bin distribution and expected availability. Its recorded complexity is O(N). Local devices then sample their response from the relevant SOC bin; the approximation is sensitive to correlation and to changes in the response law.'),
('Virtual battery. ','The fleet is represented by an equivalent energy and power envelope. The recorded complexity is O(N), because the envelope is mapped back to device-level headroom using fixed power weights. The model is compact at the operator interface, while stale envelopes can produce dataset-dependent clipping.'),
('Packetized Energy Management. ','Each device submits an indivisible 15-minute fixed-power packet. The aggregator accepts complete packets and an accepted packet remains active until expiry. The experiment definition records O(N log N). Packet granularity and asynchronous expiry introduce tracking and delay variability.'),
('Transactive control. ','Devices submit bids derived from SOC and load variation; the market clears every 15 minutes using a discrete price, and devices respond independently while that price is held. The recorded complexity is O(N log N). Performance depends on bid construction, price quantization and clearing convergence.'),
('EPS broadcast. ','The EPS implementation uses an offline Dual Quantile neural-network response estimator and a single broadcast command during routine operation. Its recorded online complexity is O(1) with respect to N. Devices retain local packet-loss, offline and Bernoulli-response mechanisms, while aggregate feedback is used for the tested adaptation and audit procedures.'),
('Centralized greedy upper bound. ','The reference scans each device headroom and performs one linear greedy allocation, with recorded complexity O(N). It represents the theoretical device-side absorption ceiling and does not apply a communication budget or network post-clipping in its upper-bound calculation. It is therefore a benchmark for interpretation rather than a deployable controller.')]

ALG_ZH=[('无协调。','该实现不发送协调信号，算子侧通信复杂度为 O(0)，设备沿用未协调基线，作为协调效果的低阶参照。'),('本地 SOC 规则。','每一步根据 s_{i,t}、a_{i,t} 和 P_i^max 在设备本地执行阈值规则，代码中复杂度为每设备 O(1)。它不使用群体反馈，结果受阈值和可用性异质性影响。'),('MPC。','滚动控制使用 H=12 个未来时段、数据驱动可用率预测和馈线剩余容量预算，实验定义中的复杂度为 O(HN)。它可以处理更多设备和网络约束，但需要状态预测和重复优化。'),('均值场控制。','实现维护 K=10 个 SOC 分箱，每 15 分钟根据预测分布和期望可用率更新统一参与概率，代码记录复杂度为 O(N)。设备在本地按分箱随机响应，强相关或响应规律变化会带来近似误差。'),('虚拟电池。','群体由等效能量和功率包络表示，按固定功率权重映射回设备，实验记录复杂度为 O(N)。接口维度较低，但过期包络会产生数据集相关的截断。'),('分组能量管理。','设备提交不可拆分的 15 分钟固定功率 packet，聚合器只接受完整 packet，接受后直到到期保持有效，实验定义复杂度为 O(N log N)。packet 粒度和异步到期会引入跟踪与延迟波动。'),('交易式控制。','设备根据 SOC 和负荷波动提交 bid，市场每 15 分钟以离散价格清算，价格保持期间设备独立响应，记录复杂度为 O(N log N)。性能取决于 bid、价格量化和清算收敛。'),('EPS 广播。','EPS 使用离线 Dual Quantile 神经网络响应估计器，日常运行发送单一广播命令，记录的在线复杂度相对于 N 为 O(1)。设备侧保留 packet loss、offline 和 Bernoulli response，聚合反馈用于测试中的适应和审计。'),('集中式贪心上限。','该参考逐设备扫描可用 headroom 并进行一次线性贪心分配，记录复杂度为 O(N)。它表示设备侧理论吸收上限，在上限计算中不设置通信预算，也不执行网络事后截断，因此仅用于解释结果，不是可部署控制器。')]

def add_alg(d,lang):
    para(d,'The following paragraphs describe the implemented logic and the complexity values used in the experiment code. The symbols refer to the standardized resource variables defined in Table S1.','Algorithm-specific interpretation. ' if lang=='en' else '算法逐项说明。')
    for lead,txt in (ALG_EN if lang=='en' else ALG_ZH): para(d,txt,boldlead=lead)
    if lang=='en':
        para(d,'The benchmark comparison is interpreted at the information boundary used by the experiment. O(1) identifies a scalar operator-side broadcast step, whereas device-side local actions are retained in the implementation. The centralized greedy upper bound is intentionally separated from the deployable methods because it assumes complete device headroom information and omits the communication constraint.')
    else:
        para(d,'算法比较遵循实验代码中的信息边界。O(1) 表示算子侧标量广播步骤，设备本地动作仍在实现中执行。集中式贪心上限假设获得全部设备 headroom，并省略通信约束，因此与可部署方法分开解释。')

def finalize(path,lang):
    d=Document(path)
    # Keep the title and table captions distinct; set all prose paragraphs to manuscript style.
    for p in d.paragraphs:
        for r in p.runs:
            if p.alignment != WD_ALIGN_PARAGRAPH.CENTER or p.text:
                runfmt(r,10,r.bold)
    # Add native equations and detailed algorithm prose before the source index table.
    # Equations are appended after S5 prose, before the URL index paragraph/table.
    idx=None
    for i,p in enumerate(d.paragraphs):
        if 'Data-source index' in p.text or '数据来源索引' in p.text: idx=i; break
    if idx is None: idx=len(d.paragraphs)
    # python-docx cannot insert before an arbitrary paragraph without XML movement; append in document order is acceptable for SI prose.
    if lang=='en':
        para(d,'The three network scales share the same radial power-flow variables and feasibility logic. Device responses p_{i,t} are assigned through z_i; the feeder solver computes branch flows, bus voltages and transformer loading before the requested response is admitted. This common layer allows the population statistic R² to be evaluated separately from the physically accepted response.','Network realization. ')
    else:
        para(d,'三个网络规模共享相同的径向潮流变量和可行性逻辑。设备响应 p_{i,t} 通过 z_i 映射到母线；馈线求解器计算支路潮流、母线电压和变压器负载后，再决定请求响应是否被接受。这一共同层使群体统计量 R² 与物理接受响应可以分开评价。','网络实现。 ')
    eq(d,'S_l = ( P_l^2 + Q_l^2 )^{1/2}   L_l = S_l / S_l^{max}   T = T_f / T^{max}')
    eq(d,'V_{min} <= V_k <= V_{max}   A_{net} = P_{acc} / P_{req}   C_{loss} = 1 - A_{net}')
    if lang=='zh':
        eq(d,'P_t = Σ_i p_{i,t}   Q_t = Σ_i q_{i,t}   N_{src} = |{ source_i }|')
        eq(d,'τ = 300 s   H = 12   R^2_{min} = 0.95   p_{ctrl,min} = 0.90   N_{eff}^* = 500')
        eq(d,'p_{i,t} = π_i ( b_t , x_{i,t} )   b_t = f ( r_t , P_t^{agg} )')
    add_alg(d,lang)
    d.save(path)

if __name__=='__main__':
    finalize(ROOT/'outputs/Supplementary_Tables_S1_S4_S5_English.docx','en')
    finalize(ROOT/'outputs/Supplementary_Tables_S1_S4_S5_Chinese.docx','zh')
