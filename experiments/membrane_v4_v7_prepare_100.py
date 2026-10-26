"""Prepare one complete 100-Z Membrane stack for frozen v4/v7 local11 inference.

This is a small CLI wrapper around the already verified CF preparation routine.
It changes only input/output paths; its sampling, percentiles and manifests stay
identical. No labels or existing predictions are read or changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import membrane_v4_v7_cf100_prepare as prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.cell / "Memb.tif").is_file() or not (args.cell / "fluo.tif").is_file():
        raise FileNotFoundError("Both Memb.tif and fluo.tif are required")
    prepare.CELL = args.cell.resolve()
    prepare.OUT = args.out.resolve()
    prepare.main()


if __name__ == "__main__":
    main()
