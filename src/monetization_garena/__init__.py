"""Garena Brazil gun-skin monetization analysis toolkit.

The reusable EDA lives in :mod:`monetization_garena.eda`. Run it with::

    python scripts/run_garena_eda.py
"""

from .eda import (  # noqa: F401
    DATA_FILE,
    FIG_DIR,
    TABLE_DIR,
    load_and_clean,
    run,
    set_style,
)

__all__ = [
    "DATA_FILE",
    "FIG_DIR",
    "TABLE_DIR",
    "load_and_clean",
    "run",
    "set_style",
]
