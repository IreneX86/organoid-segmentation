"""Read-only integrity check for a downloaded Fluo-only ER student result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(archive_path: Path, expected_sha256: str) -> dict:
    if sha256(archive_path) != expected_sha256:
        raise ValueError("Downloaded results archive SHA256 mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        prefix = "results/student/"
        required = {prefix + item for item in (
            "er_fluo_student.keras", "training_history.csv", "per_slice_metrics.csv",
            "dataset_manifest.json", "config.json", "environment.json",
            "metrics_summary.json",
        )} | {"train.log"}
        if not required.issubset(names):
            raise ValueError(f"Missing result files: {sorted(required - names)}")
        checkpoint = archive.getmember(prefix + "er_fluo_student.keras")
        if checkpoint.size <= 0:
            raise ValueError("Empty checkpoint")
        manifest = json.load(archive.extractfile(prefix + "dataset_manifest.json"))
        summary = json.load(archive.extractfile(prefix + "metrics_summary.json"))
        history = list(csv.DictReader(archive.extractfile(prefix + "training_history.csv").read().decode().splitlines()))
        slices = list(csv.DictReader(archive.extractfile(prefix + "per_slice_metrics.csv").read().decode().splitlines()))
        expected = sum(r["shape_zyx"][0] for r in manifest["records"]
                       if r["split"] in ("validation", "test"))
        if len(slices) != expected or not history or summary["epochs_completed"] != len(history):
            raise ValueError("Metrics/history counts do not match the split")
        if {r["split"] for r in slices} != {"validation", "test"}:
            raise ValueError("Missing validation or test metrics")
        if b"ER_FLUO_STUDENT_COMPLETE" not in archive.extractfile("train.log").read():
            raise ValueError("Training completion marker missing")
    return {"status": "PASS", "archive": str(archive_path), "archive_sha256": expected_sha256,
            "checkpoint_bytes": checkpoint.size, "epochs": len(history),
            "evaluated_slices": len(slices), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.sha256), indent=2))


if __name__ == "__main__":
    main()
