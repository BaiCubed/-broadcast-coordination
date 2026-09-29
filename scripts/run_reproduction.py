#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PUBLIC = ROOT / 'datasets' / 'public_datasets'
RESULTS = ROOT / 'results'
OUTPUTS = ROOT / 'outputs'
LOGS = ROOT / 'logs' / 'reproduction'
POPULATION = 'src.extra.population_experiments.run'
POPULATION_CONFIGS = 'src/extra/population_experiments/configs'
NETWORK = 'src.extra.ieee33_device_day_simulation.network_experiments'
SEEDS = '30'

def run(cmd: list[str], *, name: str, cwd: Path = ROOT, env=None):
    LOGS.mkdir(parents=True, exist_ok=True); log = LOGS / f'{name}.log'
    fullenv = os.environ.copy(); fullenv['PYTHONPATH'] = str(ROOT) + os.pathsep + fullenv.get('PYTHONPATH', '')
    if env: fullenv.update(env)
    print('RUN', ' '.join(map(str, cmd)), flush=True)
    with log.open('w', encoding='utf-8') as h:
        p = subprocess.run(cmd, cwd=cwd, env=fullenv, stdout=h, stderr=subprocess.STDOUT, text=True)
    if p.returncode: raise SystemExit(f'{name} failed with exit code {p.returncode}; see {log}')

def module(name: str, *args: str) -> list[str]:
    return [sys.executable, '-m', name, *args]

def network(name: str, *args: str) -> list[str]:
    return module(f'{NETWORK}.{name}', *args)

def population(config: str, *experiments: str) -> list[str]:
    return module(POPULATION, '--protocol', f'{POPULATION_CONFIGS}/{config}', '--experiments', *experiments)

def ensure_layout():
    RESULTS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)

def stage_check():
    run([sys.executable, 'scripts/check_mixed_populations.py'], name='check_mixed_populations')
    run([sys.executable, 'scripts/check_english_docs.py', '--root', str(ROOT)], name='check_english_docs')
    run([sys.executable, '-m', 'compileall', '-q', 'src', 'experiments', 'tools', 'scripts', 'figures'], name='compileall')

def stage_data():
    run([sys.executable, 'tools/data/preprocess.py', '--data-root', str(DATA), '--config', str(PUBLIC / 'preprocessing_config.json')], name='preprocess')
    run(module('src.extra.dataset_combinations', 'all', '--data-root', str(DATA), '--output-root', str(DATA / 'dataset_combinations')), name='dataset_combinations')
    run([sys.executable, 'tools/data/prepare_dataset_configs.py', '--all'], name='prepare_dataset_configs')

def stage_population():
    for config, experiments in (
            ('E1_population_scale.yaml', ('E1',)),
            ('E2_controller_synchronization.yaml', ('E2',)),
            ('phase_coherence.yaml', ('phase_coherence',)),
            ('E4_controller_drift.yaml', ('E4',)),
            ('response_mechanisms_and_capacity_concentration.yaml', ('response_mechanisms', 'capacity_concentration')),
            ('structured_availability.yaml', ('structured_availability',)),
            ('availability_second_family.yaml', ('availability_second_family',)),
            ('long_horizon_state.yaml', ('long_horizon_state',))):
        run(population(config, *experiments), name=config.removesuffix('.yaml'))

def stage_network_inputs():
    run(population('E1_population_scale_extended.yaml', 'E1'), name='E1_population_scale_extended')
    run(module('src.extra.dataset_experiment.run', '--all'), name='dataset_experiment')
    run(module('src.extra.dataset_experiment.network_baseline'), name='dataset_network_stress')
    run(module('src.extra.dataset_experiment.run_constrained_experiment'), name='dataset_network_constrained')
    run(network('network_dispatch_protocol'), name='network_dispatch_protocol')

def stage_network():
    steps = [
        ('train_pooled_model', ()),
        ('transfer_across_datasets', ()),
        ('transfer_alternative_splits', ()),
        ('train_mixed_model', ()),
        ('pairwise_curtailment', ()),
        ('effective_scale_mixed', ()),
        ('pairwise_predictability', ()),
        ('effective_scale_published', ()),
        ('mixed_reference_controllers', ()),
        ('ieee69_network_implementation', ('--seed-count', SEEDS, '--bootstrap-draws', '4000')),
        ('ieee69_trained_vs_transferred', ()),
        ('ieee33_trained_model', ('--seed-count', SEEDS)),
        ('ieee123_trained_model', ()),
        ('ieee123_safety_audit', ('--model-source', str(RESULTS / 'ieee123_safety_audit/trained_eps_ieee123_direct/models'), '--workers', '2', '--seed-count', SEEDS, '--bootstrap-draws', '4000')),
        ('constraint_sweep', ('--workers', '6', '--seed-count', SEEDS)),
        ('constraint_sweep_transformer', ()),
        ('constraint_sweep_transformer_summary', ()),
        ('network_stress_boundary', ('--seed-count', SEEDS, '--bootstrap-draws', '2000')),
        ('spatial_heterogeneity', ()),
        ('spatial_heterogeneity_trained_model', ()),
        ('feeder_comparison_pairwise', ('--workers', '2', '--seed-count', SEEDS)),
        ('feeder_comparison_references', ('--workers', '2', '--seed-count', SEEDS)),
    ]
    for name, args in steps: run(network(name, *args), name=name)

def stage_figures():
    run(module('figures.make_fig5'), name='make_fig5')
    run(module('figures.make_supplementary_network_figures'), name='make_supplementary_network_figures')

STAGES = ['check', 'data', 'population', 'network-inputs', 'network', 'figures']

def main():
    ap = argparse.ArgumentParser(description='Run the raw-data-to-figure reproduction chain of Figure 5 and Supplementary Figs. S24-S41.')
    ap.add_argument('--stage', choices=[*STAGES, 'all'], default='all'); args = ap.parse_args(); ensure_layout()
    if args.stage in ('check', 'all'): stage_check()
    if args.stage in ('data', 'all'): stage_data()
    if args.stage in ('population', 'all'): stage_population()
    if args.stage in ('network-inputs', 'all'): stage_network_inputs()
    if args.stage in ('network', 'all'): stage_network()
    if args.stage in ('figures', 'all'): stage_figures()
    print(json.dumps({'status': 'completed', 'stage': args.stage, 'results_root': str(RESULTS.relative_to(ROOT)), 'figures_root': 'figures/out'}, indent=2))

if __name__ == '__main__': main()
