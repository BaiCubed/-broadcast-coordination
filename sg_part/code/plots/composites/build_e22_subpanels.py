"""Build E22 subpanels from their independent leaf builders."""
from tools import generate_e22_subpanel_figures as figures


def main() -> None:
    figures.OUT.mkdir(parents=True, exist_ok=True)
    for builder in figures.PLOTS.values():
        builder()
    figures.write_readme()


if __name__ == "__main__":
    main()
