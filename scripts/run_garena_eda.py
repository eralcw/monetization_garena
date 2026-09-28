"""Entry point for the Garena Brazil gun-skin EDA.

Run from the repository root::

    python scripts/run_garena_eda.py

Figures are written to ``reports/figures`` and summary tables to
``reports/tables``. All figures use English labels and the Garena palette.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

# Make the ``src`` package importable without installing it.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from monetization_garena.eda import run  # noqa: E402

if __name__ == "__main__":
    run()
