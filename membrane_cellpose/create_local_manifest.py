"""Record hashes of the local Membrane files without copying biological data.

Run with ``python membrane_cellpose/create_local_manifest.py /path/to/Membrane``.
The JSON contains relative paths, sizes and SHA256 digests only. It does not
upload, open as images, or change any TIFF, annotation, prediction or model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Local Membrane directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        parser.error(f"Not a directory: {source}")

    records = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name.startswith("._"):
            continue
        records.append({
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_label": "Membrane (local; contents not uploaded)",
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(records)} files, {manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()
