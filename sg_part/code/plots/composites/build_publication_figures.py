"""Build the publication figures by invoking one leaf plot at a time."""
from tools.generate_boxplot_outputs import PLOTS, finish_summary


def main() -> None:
    for name, builder in PLOTS.items():
        builder()
    finish_summary()


if __name__ == "__main__":
    main()
