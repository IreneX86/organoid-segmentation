"""Verify one downloaded 100-Z v4/v7 archive before deleting its Pod."""

from __future__ import annotations

import argparse
from pathlib import Path

import membrane_v4_v7_cf100_verify as verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    verify.ROOT = args.root.resolve()
    verify.main()


if __name__ == "__main__":
    main()
