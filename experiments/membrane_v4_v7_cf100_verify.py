"""Verify the downloaded CF inference archive before Pod deletion.

Rejects path traversal and pre-existing extraction targets, checks archive and
each full-stack TIFF SHA256, source/model provenance, GPU preflight and shape.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import tifffile

from membrane_v4_base_v7_additions_20261023 import sha256


ROOT = Path(__file__).resolve().parents[1] / "runpod_results/membrane_v4_v7_cf100_20261023"


def main() -> None:
    archive = ROOT / "results.tar.gz"
    expected = (ROOT / "results.tar.gz.sha256").read_text().split()[0]
    if sha256(archive) != expected:
        raise RuntimeError("Downloaded archive SHA256 mismatch")
    out = ROOT / "downloaded"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if member.name.startswith("/") or ".." in parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsafe archive entry: {member.name}")
        stream.extractall(out, filter="data")
    if (out / "exit_status.txt").read_text().strip() != "0":
        raise RuntimeError("Remote inference exited nonzero")
    log = (out / "inference.log").read_text(errors="replace")
    if "GPU_PREFLIGHT_PASS" not in log or "MEMBRANE_CF100_V4_V7_COMPLETE" not in log or "Traceback" in log:
        raise RuntimeError("Missing GPU/completion marker or traceback in remote log")
    original = json.loads((ROOT / "input_manifest.json").read_text())
    fetched = json.loads((out / "input_manifest.json").read_text())
    if original != fetched:
        raise RuntimeError("Downloaded manifest differs from local frozen manifest")
    config = json.loads((out / "results/run_config.json").read_text())
    if config["visible_gpu_count"] != 1 or config["cellpose"] != "4.2.1.1":
        raise RuntimeError("Unexpected execution environment")
    for version in ("v4", "v7"):
        model_name = f"cpsam_finetuned_{version}"
        if config["model_sha256"][version] != original["files"][model_name]["sha256"]:
            raise RuntimeError(f"Wrong {version} model")
        path = out / "results" / config["runs"][version]["output"]
        if sha256(path) != config["runs"][version]["output_sha256"]:
            raise RuntimeError(f"Wrong {version} output SHA256")
        with tifffile.TiffFile(path) as tif:
            if tif.series[0].shape != (100, 1200, 1200) or tif.series[0].dtype.name != "uint16":
                raise RuntimeError(f"Wrong {version} TIFF geometry/dtype")
    record = {"result": "PASS", "archive_sha256": expected,
              "gpu": config["gpu"], "3d_ids": {name: config["runs"][name]["3d_ids"]
                                            for name in ("v4", "v7")},
              "outputs": {name: str(out / "results" / config["runs"][name]["output"])
                          for name in ("v4", "v7")}}
    (ROOT / "local_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
