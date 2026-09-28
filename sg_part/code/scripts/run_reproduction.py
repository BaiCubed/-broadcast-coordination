#!/usr/bin/env python3
"""Run the complete raw-data-to-output reproduction chain.

Stages are explicit because E20-E24 can require many hours and large temporary
storage. The default ``all`` stage runs every stage in dependency order.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

CODE=Path(__file__).resolve().parents[1]
RELEASE=CODE.parent
DATA=RELEASE/'data'
RESULTS=CODE/'results'
OUTPUTS=CODE/'outputs'
LOGS=CODE/'reproduction_logs'

def run(cmd:list[str], *, name:str, cwd:Path=CODE, env=None):
    LOGS.mkdir(exist_ok=True); log=LOGS/f'{name}.log'
    fullenv=os.environ.copy(); fullenv['PYTHONPATH']=str(CODE)+os.pathsep+fullenv.get('PYTHONPATH','')
    if env: fullenv.update(env)
    print('RUN', ' '.join(map(str,cmd)), flush=True)
    with log.open('w',encoding='utf-8') as h:
        p=subprocess.run(cmd,cwd=cwd,env=fullenv,stdout=h,stderr=subprocess.STDOUT,text=True)
    if p.returncode: raise SystemExit(f'{name} failed with exit code {p.returncode}; see {log}')

def ensure_layout():
    link=CODE/'data'
    if link.exists() and not link.is_symlink(): raise SystemExit(f'{link} exists and is not the release data link')
    if not link.exists(): link.symlink_to(DATA, target_is_directory=True)
    (CODE/'results').mkdir(exist_ok=True); (CODE/'outputs').mkdir(exist_ok=True)

def stage_check():
    run([sys.executable,'scripts/check_release.py'],name='check_release')
    run([sys.executable,'scripts/check_english_docs.py'],name='check_english_docs')
    reference = RELEASE.parent/'outputs'
    if reference.is_dir():
        run([sys.executable,'scripts/build_output_manifest.py','--reference-root',str(reference)],name='output_manifest')
    else:
        print('reference outputs tree not bundled; using the checked-in manifest')
    run([sys.executable,'-m','compileall','-q','src','experiments','tools','generation'],name='compileall')

def stage_data():
    # Canonical caches are generated from the official-source directories.
    run([sys.executable,'generation/preprocess.py','--data-root',str(DATA)],name='preprocess')
    run([sys.executable,'-m','src.extra.dataset_combinations','all','--data-root',str(DATA),'--output-root',str(DATA/'dataset_combinations')],name='dataset_combinations')

def stage_e1_e4():
    run([sys.executable,'-m','tools.prepare_dataset_configs','--all'],name='prepare_dataset_configs')
    run([sys.executable,'-m','src.extra.nc_excel_experiments.run','--protocol','src/extra/nc_excel_experiments/configs/protocol.yaml','--experiments','E1','E2','E3','E4'],name='e1_e4')

def stage_e20_e24():
    # The runners reuse completed checkpoints when present and write manifests.
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e20_transfer'],name='e20')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e21_mixed_scenarios'],name='e21_mixed')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e21_pairwise_curtailment'],name='e21_pairwise_curtailment')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e21_pairwise_r2'],name='e21_pairwise_r2')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e21_gamma_universal_boundary'],name='e21_gamma')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e21_mixed_gamma_boundary'],name='e21_mixed_gamma')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e22_ieee69_complexity'],name='e22')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e23_ieee69_relative_boundary'],name='e23')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e23_spatial_heterogeneity'],name='e23_spatial')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_direct_training'],name='e24_training')
    run([sys.executable,'-m','src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_audit','--model-source',str(RESULTS/'E24/trained_eps_ieee123_direct/models')],name='e24_audit')

def stage_artifacts():
    commands=[
      ('boxplots',['tools/generate_boxplot_outputs.py']),
      ('replot_results',['tools/replot_results_e20_e24.py']),
      ('subpanel_figures',['tools/generate_e22_subpanel_figures.py']),
      ('subpanel_presentation',['tools/build_e22_subpanel_presentation.py']),
      ('appendix',['tools/build_appendix.py']),
      ('supplementary_tables',['tools/build_supplementary_tables_s1_s4_s5.py']),
      ('mixing_report',['tools/build_mixing_methods_report.py']),
      ('e23_e24_outputs',['tools/build_e23_e24_outputs.py']),
    ]
    for name,script in commands: run([sys.executable,*script],name=name)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['check','data','e1-e4','e20-e24','artifacts','all'],default='all'); args=ap.parse_args(); ensure_layout()
    if args.stage in ('check','all'): stage_check()
    if args.stage in ('data','all'): stage_data()
    if args.stage in ('e1-e4','all'): stage_e1_e4()
    if args.stage in ('e20-e24','all'): stage_e20_e24()
    if args.stage in ('artifacts','all'): stage_artifacts()
    print(json.dumps({'status':'completed','stage':args.stage,'results_root':str(RESULTS),'outputs_root':str(OUTPUTS)},indent=2))
if __name__=='__main__': main()
