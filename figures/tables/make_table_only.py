import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'make_supp_tables_2-4.py'), encoding='utf-8').read()
MARK = '# ---------------------------------------------------------------- Table 2'
head, body = SRC.split(MARK)
body = body.replace("doc.save(OUT)\nprint('saved', OUT)\n", '')

exec(head)
OUT = os.path.join(HERE, 'Supplementary_Tables.docx')
note = lambda text: None

caption(1, 'Published populations and constructed mixed populations.')
part('a,', 'Published populations')
table(['Resource class', 'Population', 'Native time step', 'Available independent sources',
       'Largest simulated size *N*', 'Size levels simulated'], [
    ['Distribution feeder', 'LV urban 35297', '1 h', '12,000', '3,000', '18'],
    [None, 'LV urban 8087', '1 h', '4,998', '3,000', '18'],
    [None, 'LV rural 2731', '1 h', '2,730', '2,730', '18'],
    ['Household meter', 'Goiener', '1 h', '4,998', '3,000', '18'],
    [None, 'Low Carbon London', '30 min', '4,998', '3,000', '18'],
    [None, 'Smart Grid Smart City', '30 min', '4,998', '3,000', '18'],
    [None, 'Norway AMI', '1 h', '2,982', '2,982', '18'],
    [None, 'Irish smart meters', '30 min', '2,898', '2,898', '18'],
    [None, 'CAMSL Japan', '30 min', '1,422', '1,422', '15'],
    [None, 'OPSD households', '15 min', '6', '6', '5'],
    ['Thermal load', 'Danish heat meters', '1 h', '2,400', '2,400', '17'],
    [None, 'HEAPO heat pumps', '15 min', '1,338', '1,338', '15'],
    ['Building and community', 'Building Data Genome 2', '1 h', '1,566', '1,566', '15'],
    [None, 'Building Data Genome 1', '1 h', '504', '504', '13'],
    [None, 'Energy community', '15 min', '246', '246', '11'],
], [3.0, 3.8, 2.3, 2.8, 2.8, 2.3], merges=[(0, 0, 2, 0), (3, 0, 9, 0), (10, 0, 11, 0), (12, 0, 14, 0)])

part('b,', 'Constructed mixed populations')
table(['Mixture type', 'Number', 'Composition', 'Sources per mixture', 'Weights'], [
    ['Pairwise', '105', 'Every pair of the 15 published populations', '2', '0.5 each'],
    ['Designed multi-source', '2', 'Group 1: three similar household-meter sources; building + heat pump + household meter',
     '3', 'Equal'],
    [None, '2', 'Group 2: LV feeders with smart meters; building, household, thermal and community sources',
     '5', 'Unequal (largest 0.35–0.40)'],
    [None, '2', 'Group 3: ten electricity sources; ten sources across sectors', '10', 'Unequal'],
    [None, '2', 'Group 4: all 15 populations, weighted equally by population or by resource type',
     '15', 'Equal or type-balanced'],
    [None, '3', 'Group 5: long-tailed mixtures dominated by household, building–thermal or feeder–community sources',
     '15', 'Unequal'],
    [None, '3', 'Group 6: spatially zoned mixtures (LV feeders; cross-sector; community and two-way metered sources)',
     '4–5', 'Unequal; each source assigned to feeder zones'],
], [3.0, 1.5, 7.2, 2.2, 3.1], merges=[(1, 0, 6, 0)], left_cols=(2,))

doc.add_page_break()

exec(body)

doc.add_page_break()

BODY_PT = 8.5
caption(5, 'Distribution-feeder models and network evaluation settings.')
part('a,', 'Feeder models')
table(['Feeder', 'Buses / branches', 'Base voltage', 'Voltage limits', 'Substation transformer',
       'Branch thermal limits', 'Power-flow model', 'Used in'], [
    ['Reference feeder (IEEE-33 configuration)', '33 / 32', '12.66 kV', '0.95–1.05 p.u.', '3,000 kW',
     'Fixed, 250–500 kW per branch', 'Linear voltage sensitivity; network feedback off in scale runs',
     'Population-scale experiments; reference case in Supplementary Note 8'],
    ['IEEE-33 (Baran–Wu)', '33 / 32', '12.66 kV', '0.95–1.05 p.u.', '6,000 kVA',
     'From downstream base load (≥90 kVA)', 'LinDistFlow', 'Result 4 (M0–M6); Supplementary Note 8'],
    ['IEEE-69', '69 / 68', '12.66 kV', '0.95–1.05 p.u.', '6,000 kVA',
     'From downstream base load (≥70 kVA)', 'LinDistFlow', 'Result 4 (M0–M6, constraint sweeps); Supplementary Note 8'],
    ['IEEE-123 (single-phase radial equivalent)', '124 / 123', '4.16 kV', '0.95–1.05 p.u.', '5,000 kVA',
     'From downstream base load (≥70 kVA)', 'LinDistFlow', 'Result 4 (placement tests); Supplementary Note 8'],
], [2.6, 1.5, 1.4, 1.7, 1.9, 2.4, 2.5, 3.0])

part('b,', 'Network operating conditions (IEEE-33 and IEEE-69)')
table(['Condition', 'Load', 'Solar', 'Wind', 'Branch and transformer capacity', 'Forecast error',
       'Location of renewable input and storage devices'], [
    ['M0 Nominal', '0.45', '0.75', '0.30', '1.00', '0', 'Input proportional to load; devices placed by load'],
    ['M1 Deep-feeder stress', '0.56', '0.85', '0.35', '0.68', '0', 'As M0'],
    ['M2 Reverse-flow bottleneck', '0.48', '1.00', '0.40', '0.72', '0', 'Input concentrated at two distal groups'],
    ['M3 Derating and error', '0.56', '1.05', '0.45', '0.65; three branches further derated to 0.55–0.65', '0.18',
     '90% of input clustered'],
    ['M4 Uniform placement', 'As M0', 'As M0', 'As M0', 'As M0', 'As M0', 'Devices spread uniformly over buses'],
    ['M5 50% feeder concentration', 'As M0', 'As M0', 'As M0', 'As M0', 'As M0', '50% of devices on one distal lateral'],
    ['M6 80% node concentration', 'As M0', 'As M0', 'As M0', 'As M0', 'As M0', '80% of devices at one distal bus'],
], [3.2, 1.1, 1.1, 1.1, 3.3, 1.4, 5.8], left_cols=(6,))

part('c,', 'Placement and capacity tests (IEEE-123)')
table(['Test', 'Device placement', 'Branch and transformer capacity'], [
    ['Uniform', 'Spread uniformly over buses', '1.0'],
    ['Distal lateral', '50% of devices on one distal lateral', '1.0'],
    ['Distal bus', '80% of devices at one distal bus', '1.0'],
    ['Distal lateral, reduced capacity', '50% of devices on one distal lateral', '0.7'],
], [5.0, 7.0, 5.0])

part('d,', 'Constraint sweeps (IEEE-69)')
table(['Swept quantity', 'Range', 'What changes'], [
    ['Request intensity *q*', '0–1.5, step 0.1', 'Size of the requested aggregate response'],
    ['Line pressure *λ*', '0–0.40, step 0.05', 'Branch and transformer capacity scaled to 1 − *λ*'],
    ['Spatial concentration *κ*', '0–0.8, step 0.1', 'Degree to which devices are concentrated in one part of the feeder'],
], [4.0, 3.5, 9.5], left_cols=(2,))

doc.save(OUT)
print('saved', OUT)
