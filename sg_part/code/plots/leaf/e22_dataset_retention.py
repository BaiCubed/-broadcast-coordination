"""Build the E22 dataset-retention subpanel."""
from tools.generate_e22_subpanel_figures import PLOTS

if __name__ == "__main__":
    PLOTS["05_ieee69_dataset_algorithm_retention"]()
