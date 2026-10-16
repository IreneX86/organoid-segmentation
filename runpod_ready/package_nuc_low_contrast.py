"""Package the frozen Nuc data, code, and baseline for one low-contrast run.

The raw TIFFs and reference checkpoint are read only. Their SHA256 digests are
checked before they enter a Git-ignored transfer archive; nothing is edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


BASELINE_SHA256 = "c1191f00e327d6f73acc73f202a3a8e44895cb669bbfa2c6a1f3f664cc986b9e"
FILES = (
    "code/nuc_train_internal.py",
    "code/nuc_low_contrast_stress.py",
    "code/evaluation.py",
    "code/losses.py",
    "code/model/__init__.py",
    "code/model/unet.py",
    "code/patch_dataset.py",
    "code/pipeline_config.py",
    "code/segmentation_metrics.py",
    "code/sliding_window.py",
    "code/tiff_io.py",
    "code/requirements.txt",
    "runpod_ready/setup_env.sh",
    "runpod_ready/nuc_gpu_preflight.py",
    "runpod_ready/nuc_low_contrast_run.sh",
    "experiments/nuc_low_contrast_protocol_20261016.md",
    "experiments/results/nuc_internal_split_v1.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.archive.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.archive}")
    split_path = args.code_root / "experiments/results/nuc_internal_split_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    names = sorted(set(split["train"] + split["validation"]))
    if (len(names) != 167 or len(split["train"]) != 133
            or len(split["validation"]) != 34):
        raise ValueError("Expected frozen 167/133/34 Nuc split")

    if sha256(args.baseline_model) != BASELINE_SHA256:
        raise ValueError("Frozen baseline checkpoint SHA256 differs")
    entries = [(args.code_root / relative, relative) for relative in FILES]
    entries.append((args.baseline_model, "reference/nuc_baseline.keras"))
    for subdir, hash_key in (
        ("Nuc_input", "input_sha256"),
        ("Nuc_output", "label_sha256"),
    ):
        for name in names:
            path = args.data_root / "label_ref" / "Nuc" / subdir / name
            if sha256(path) != split[hash_key][name]:
                raise ValueError(f"Frozen source hash mismatch: {subdir}/{name}")
            entries.append((path, f"label_ref/Nuc/{subdir}/{name}"))

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.archive.with_name(args.archive.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite partial archive: {temporary}")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
            for source, relative in entries:
                if not source.is_file() or source.is_symlink():
                    raise ValueError(f"Expected an ordinary file: {source}")
                archive.add(source, arcname=f"Organoid/{relative}", recursive=False)
        temporary.replace(args.archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256(args.archive)
    checksum = args.archive.with_name(args.archive.name + ".sha256")
    checksum.write_text(f"{digest}  {args.archive.name}\n", encoding="ascii")
    print(json.dumps({
        "archive": str(args.archive), "sha256": digest,
        "members": len(entries), "pairs": len(names), "bytes": args.archive.stat().st_size,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
