from __future__ import annotations

import argparse
from pathlib import Path

from ..data2_correlation_paired.run_experiment import run_suite


ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = ROOT / "results" / "data2_ieee33_adapted"
CONFIG_ROOT = ROOT / "src" / "extra" / "ieee33_device_day_simulation" / "experiments" / "data2_ieee33_adapted" / "configs"

PROFILES = {
    "observed_demand": {
        "population": "population_data2_demand.yaml",
        "description": "out_power is the observed charging demand; no external supply is asserted.",
        "external_input": "none",
        "load": "observed_out_power",
    },
    "input_counterfactual": {
        "population": "population_data2_input_counterfactual.yaml",
        "description": "out_power replaces the Solar channel as an explicitly counterfactual input.",
        "external_input": "observed_out_power",
        "load": "observed_out_power",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated, configuration-driven data2 IEEE33 adapter experiment.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="observed_demand")
    parser.add_argument("--group", choices=("network_weak_data2_adapted", "network_stress_data2_adapted"), default=None)
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    groups = (
        {
            "name": "network_weak_data2_adapted",
            "config": CONFIG_ROOT / f"default_network_weak_data2_{'input_counterfactual' if args.profile == 'input_counterfactual' else 'adapted'}.yaml",
            "description": "IEEE33 weak network condition using the isolated data2 adapter.",
        },
        {
            "name": "network_stress_data2_adapted",
            "config": CONFIG_ROOT / f"default_network_stress_data2_{'input_counterfactual' if args.profile == 'input_counterfactual' else 'adapted'}.yaml",
            "description": "IEEE33 stress network condition using the isolated data2 adapter.",
        },
    )
    run_suite(
        Path(args.results_root) / args.profile,
        selected_group=args.group,
        groups=groups,
        protocol_name=f"data2_ieee33_adapted_{args.profile}",
        data2_input={
            "adapter_profile": args.profile,
            "external_energy_input": profile["external_input"],
            "charging_demand": profile["load"],
            "source_field": "out_power",
            "source_time_field": "end_time",
            "nextgen_untouched": True,
        },
    )


if __name__ == "__main__":
    main()
