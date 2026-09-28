#!/usr/bin/env python3
"""等待 baseline 补跑完成后自动更新子图和总图。"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/E22/subpanel_direct_supplement/baseline_completion_checkpoint.json"
LOG = ROOT / "logs/subpanel_baseline_completion_finalize.log"
PYTHON = Path("/home/heol/anaconda3/bin/python")


def _completed() -> bool:
    if not CHECKPOINT.is_file():
        return False
    try:
        return json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    while not _completed():
        time.sleep(60)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/mpl-subpanel-finalize"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        subprocess.run(
            [str(PYTHON), "tools/generate_e22_subpanel_figures.py"],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [str(PYTHON), "tools/build_e22_subpanel_presentation.py", "--source-dir", "outputs/figs/subpanel"],
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    main()
