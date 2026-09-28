#!/usr/bin/env python3
"""Build the artifact-level reproduction manifest from a reference outputs tree."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

def classify(rel: str) -> tuple[str,str,str]:
    if rel.startswith('data_available/'): return ('data_availability','generation/preprocess.py or generation/dataset_combinations','python generation/preprocess.py --data-root data; python -m generation.dataset_combinations all --data-root data')
    if rel.startswith('appendix/'): return ('appendix','tools/build_appendix.py','python tools/build_appendix.py')
    if rel.startswith('results-E20-E24/'): return ('source_result_migration','tools/replot_results_e20_e24.py','python tools/replot_results_e20_e24.py')
    if rel.startswith('figs/subpanel/'): return ('composite_figures','tools/generate_e22_subpanel_figures.py','python tools/generate_e22_subpanel_figures.py')
    if rel.startswith('figs/direct_train/source_data/') or rel.startswith('figs/direct_train/03_') or rel.startswith('figs/direct_train/04_'): return ('composite_figures','tools/plot_subpanel_03_04_direct_supplement.py','python tools/plot_subpanel_03_04_direct_supplement.py')
    if rel.startswith('figs/direct_train/07_') or rel.startswith('figs/direct_train/e22_grid_constrained_panels_'): return ('composite_figures','tools/build_e22_subpanel_presentation.py','python tools/build_e22_subpanel_presentation.py')
    if rel.startswith('figs/direct_train/') and not rel.endswith('README_03_04_SUPPLEMENT.md'): return ('composite_figures','tools/plot_direct_train_subpanel.py','python tools/plot_direct_train_subpanel.py')
    if rel.startswith('figs/e22_ieee123_appendix/'): return ('appendix','tools/build_e22_ieee123_appendix_figures.py','python tools/build_e22_ieee123_appendix_figures.py')
    if rel.startswith('figs/e22_ieee69_appendix/'): return ('appendix','tools/build_e22_constraint_component_heatmap.py','python tools/build_e22_constraint_component_heatmap.py')
    if rel.startswith('figs/e22_grid_constrained_panels'): return ('composite_figures','tools/build_e22_subpanel_presentation.py','python tools/build_e22_subpanel_presentation.py')
    if rel.startswith('figs/supplementary_s1_s4_s5/'): return ('supplementary_figures','tools/build_supplementary_tables_s1_s4_s5.py','python tools/build_supplementary_tables_s1_s4_s5.py')
    if rel.startswith('mixing_methods_figures/') or rel == 'data_mixing_methods_report.docx': return ('report','tools/build_mixing_methods_report.py','python tools/build_mixing_methods_report.py')
    if rel.startswith(('00-','01-','02-','03-','04-','05-','06-','07-','08-')): return ('publication_figures','tools/generate_boxplot_outputs.py','python tools/generate_boxplot_outputs.py')
    if rel.startswith('Supplementary_Tables_S1_S4_S5_') or rel == 'table example.png': return ('supplementary_tables','tools/build_supplementary_tables_s1_s4_s5.py','python tools/build_supplementary_tables_s1_s4_s5.py')
    if rel.startswith('E23_E24') or rel.startswith('results/outputs/'): return ('report','tools/build_e23_e24_outputs.py','python tools/build_e23_e24_outputs.py')
    return ('metadata_or_manual','manual_or_source_metadata','see artifact notes')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reference-root',type=Path,required=True); ap.add_argument('--output',type=Path,default=Path('audit/OUTPUT_REPRODUCTION_MANIFEST.csv')); args=ap.parse_args()
    files=sorted(p for p in args.reference_root.rglob('*') if p.is_file()); args.output.parent.mkdir(parents=True,exist_ok=True)
    fields=['artifact','release_path','bytes','sha256','stage','generator','command','reference_status','release_status']; counts={}; rows=[]
    for p in files:
        rel=p.relative_to(args.reference_root).as_posix(); stage,gen,cmd=classify(rel); counts[stage]=counts.get(stage,0)+1
        if rel.startswith('data_available/generation/'):
            release_path = 'generation/' + rel[len('data_available/generation/'):]
        elif rel.startswith('data_available/'):
            release_path = '../data/' + rel[len('data_available/'):]
        else:
            release_path = 'outputs/' + rel
        rows.append({'artifact':rel,'release_path':release_path,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'stage':stage,'generator':gen,'command':cmd,'reference_status':'present_in_reference_tree','release_status':'rebuild_from_code'})
    with args.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (args.output.with_suffix('.json')).write_text(json.dumps({'artifact_count':len(rows),'by_stage':counts},indent=2)+'\n')
    print(f'wrote {len(rows)} artifacts to {args.output}')
if __name__=='__main__': main()
