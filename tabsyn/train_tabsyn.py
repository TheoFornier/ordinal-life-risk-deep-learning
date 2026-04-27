from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_TABSYN_DIR = BASE_DIR / "external" / "tabsyn"

CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]


# Lance une commande Python dans le dossier external/tabsyn.
def run_cmd(args: list[str]) -> None:
    cmd = [sys.executable] + args
    print("\nCommande lancée :")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=EXTERNAL_TABSYN_DIR, check=True)


def main() -> None:
    for class_value in CLASSES:
        dataname = f"prudential_class_{class_value}"

        print(f"\n===== {dataname} =====")

        run_cmd(["process_dataset.py", "--dataname", dataname])
        run_cmd(["main.py", "--dataname", dataname, "--method", "vae", "--mode", "train"])
        run_cmd(["main.py", "--dataname", dataname, "--method", "tabsyn", "--mode", "train"])
        run_cmd(["main.py", "--dataname", dataname, "--method", "tabsyn", "--mode", "sample"])


if __name__ == "__main__":
    main()