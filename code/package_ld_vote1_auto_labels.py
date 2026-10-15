"""Package the reviewed 1/5 LD masks as ten separate, mask-only ZIPs.

Read only Desktop ``fluo_label_vote1`` outputs. Preflight every input, source
manifest, and mask before writing a separate Git delivery. Re-running verifies
existing packages byte-for-byte and never overwrites them or the source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import tifffile


CELL_COUNTS = {
    "cell_22375_NZ_100": 100,
    "cell_22376_NZ_100": 100,
    "cell_22377_NZ_120": 120,
    "cell_22378_NZ_150": 150,
    "cell_22379_NZ_150": 150,
    "cell_22380_150": 150,
    "cell_22381_140": 140,
    "cell_22382_140": 140,
    "cell_22383_140": 140,
    "cell_22385_140": 140,
}
METHOD = "five_fold_binary_vote_at_locked_validation_thresholds_one_of_five"
SCIENTIFIC_STATUS = "exploratory_human_review_preferred_not_independently_validated"
SETTINGS = {
    "lower_percentile": 1.0,
    "upper_percentile": 99.8,
    "patch_size": 512,
    "overlap": 128,
    "prediction_batch_size": 4,
}
# Public provenance hashes of the same five locked focal folds used for 3/5.
# The threshold and config hashes below additionally bind this 1/5 run to its
# source manifest; no checkpoint or private configuration is packaged.
FOLDS = (
    (0, 0.65, "7c73c1e9ae890e07d6a4af41a299611066b573a69e5ad7813ee7d63633a2fc1e", "3637e4e21ec776f0e13f577ab3fe3ba90781dfceea5e49dc42e2d6e3e3258045", "34b770045820c362d257b822f6c5d4f12716f7ba61e86e2cd86dd2761ddf2ef7", "879887cb418b37b7dadbab0205e1b9ca7e2103155e90b1cc7b93befbb6e47226"),
    (1, 0.80, "1c6ea7ece88ab2ab807d3171e67e025a9a6a94c2976e20961333d8775d44d614", "af278de6ee9317225ce3772bae2eba5b9d172f95f3219c064df8334b4de85902", "20f82d593f48c17f4e4a1277b1c38f825a55436a6beddfe46a3afbec690fd6cc", "3b8071b09a748a5aad2c9d07093a1ac68b6e1f1f588b8080c22fb0e71cd9948b"),
    (2, 0.60, "c4c7dbde1bb37647e546ee8690aa5f7f8b0695a3d285fb4bf91270f30140f2df", "c5ca87cb3f97f4402249be8adb6a63a27c5a697835480b811f09a9c649262c2e", "d59afe83d40a25c9c3acf7d529783dc126b3633845538f40ecb2b1fcf1785b35", "a5a9e7e00af5b471aa6dd121e4de7ad23c6f7c836c924adc82d3007ddde3ad2d"),
    (3, 0.35, "8b3674bf4d670ed4e32c8e907180b7201c16879bc6035093c02f22a4f09fa55e", "afeec0d81e7dfd15881ae2f2ae2af6d9f0ce021b08daaaba88561c2bc7efd319", "2f3b781991d4416e2658818d3166883f2cdcc5336fc7f837a13bc86fb20bd087", "6dd780e192f2d8b47babc9c86b012ded08af4c0b642f0215b8d61d82936082cb"),
    (4, 0.50, "a31d517c83ba710cc67c256455dcde09dfc46ae3cccfc474f663b191e56d413f", "fbd7f4fd0c48c75439a2e892e5ee7639948ff29067d2ed1b4c3f4f5469e5dbe6", "acb8a2371a66a67e332b70c8aab3f477319a1eb80633bd8d63d3b942308a6770", "8151ae44ce3593de64cc22d05205de2b83bab45f1b7a9e90441ade765de29a9e"),
)
ZIP_TIMESTAMP = (2026, 10, 15, 0, 0, 0)
MAX_GIT_BLOB_BYTES = 100_000_000
ARCHIVE_NAME = "fluo_label_vote1.zip"
PREFIX = "fluo_label_vote1"
SOURCE_RUN_ROOT = "/workspace/ld_vote1_bulk_20261015/bundle"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_single_tiff(path: Path, *, mask: bool) -> np.ndarray:
    with tifffile.TiffFile(path) as tiff:
        if len(tiff.pages) != 1 or len(tiff.series) != 1:
            raise ValueError(f"Expected one 2D TIFF image: {path}")
        page = tiff.pages[0]
        if mask and page.photometric != tifffile.PHOTOMETRIC.MINISBLACK:
            raise ValueError(f"Mask is not MINISBLACK: {path}")
        image = page.asarray()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF image: {path}")
    return image


def verified_folds(folds: object, cell: str) -> list[dict]:
    if not isinstance(folds, list) or len(folds) != len(FOLDS):
        raise ValueError(f"Five fold records required: {cell}")
    portable = []
    for record, (index, threshold, checkpoint, threshold_hash, lock, config) in zip(folds, FOLDS):
        if not isinstance(record, dict) or set(record) != {
            "fold", "threshold", "checkpoint_sha256", "threshold_sha256",
            "selection_lock_sha256", "config_sha256",
        }:
            raise ValueError(f"Unexpected fold record: {cell}")
        actual_threshold = float(record["threshold"])
        if (record["fold"] != index
                or not math.isfinite(actual_threshold)
                or not math.isclose(actual_threshold, threshold, rel_tol=0, abs_tol=1e-8)
                or record["checkpoint_sha256"] != checkpoint
                or record["threshold_sha256"] != threshold_hash
                or record["selection_lock_sha256"] != lock
                or record["config_sha256"] != config):
            raise ValueError(f"Locked model or threshold differs: {cell}/fold_{index}")
        portable.append({**record, "threshold": threshold})
    return portable


def verify_source(cell: str, data_root: Path, inference_code_hash: str) -> dict:
    cell_root = data_root / cell
    input_dir = cell_root / "fluo"
    mask_dir = cell_root / PREFIX
    manifest_path = mask_dir / "prediction_manifest.json"
    for folder in (cell_root, input_dir, mask_dir):
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"Missing or symlinked source folder: {folder}")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"Missing or symlinked prediction manifest: {cell}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "status": "complete",
        "target": "ld",
        "scientific_status": SCIENTIFIC_STATUS,
        "method": METHOD,
        "not_validated_as_ensemble": True,
        "input_dir": f"{SOURCE_RUN_ROOT}/inputs/{cell}/fluo",
        "output_dir": f"{SOURCE_RUN_ROOT}/outputs/{cell}/{PREFIX}",
        "settings": SETTINGS,
        "foreground_value": 255,
        "inference_code_sha256": inference_code_hash,
        "image_count": CELL_COUNTS[cell],
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError(f"Wrong or incomplete 1/5 source manifest: {cell}")
    folds = verified_folds(manifest.get("folds"), cell)
    expected = [f"fluo_{number:03d}.tif" for number in range(1, CELL_COUNTS[cell] + 1)]
    source_entries = {p.name for p in input_dir.iterdir() if not p.name.startswith("._")}
    mask_entries = {p.name for p in mask_dir.iterdir() if not p.name.startswith("._")}
    if source_entries != set(expected) or mask_entries != set(expected) | {"prediction_manifest.json"}:
        raise ValueError(f"Unexpected or missing source/mask files: {cell}")
    source_records = manifest.get("files")
    if (not isinstance(source_records, list) or len(source_records) != len(expected)
            or [r.get("filename") for r in source_records if isinstance(r, dict)] != expected):
        raise ValueError(f"Manifest image names/count/order differ: {cell}")

    records = []
    for name, record in zip(expected, source_records):
        input_path = input_dir / name
        mask_path = mask_dir / name
        if (input_path.is_symlink() or mask_path.is_symlink()
                or not input_path.is_file() or not mask_path.is_file()):
            raise ValueError(f"Missing or symlinked TIFF: {cell}/{name}")
        input_image = read_single_tiff(input_path, mask=False)
        mask_image = read_single_tiff(mask_path, mask=True)
        if (mask_image.shape != input_image.shape or mask_image.dtype != np.uint8
                or not np.isin(mask_image, (0, 255)).all()
                or record.get("height") != input_image.shape[0]
                or record.get("width") != input_image.shape[1]
                or record.get("foreground_pixels") != int(np.count_nonzero(mask_image))):
            raise ValueError(f"Mask grid, values, or count differ: {cell}/{name}")
        input_hash = sha256_file(input_path)
        mask_hash = sha256_file(mask_path)
        if record.get("input_sha256") != input_hash or record.get("mask_sha256") != mask_hash:
            raise ValueError(f"Input or mask SHA256 differs: {cell}/{name}")
        records.append({
            "filename": name,
            "height": int(mask_image.shape[0]),
            "width": int(mask_image.shape[1]),
            "foreground_pixels": int(np.count_nonzero(mask_image)),
            "input_sha256": input_hash,
            "mask_sha256": mask_hash,
        })
    return {
        "cell": cell,
        "records": records,
        "folds": folds,
        "source_manifest_sha256": sha256_file(manifest_path),
    }


def zip_entry(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def verify_archive(path: Path, records: list[dict], metadata: dict[str, bytes]) -> None:
    expected = {f"{PREFIX}/{r['filename']}": r["mask_sha256"] for r in records}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected) | set(metadata):
            raise ValueError(f"ZIP member list differs: {path}")
        if archive.testzip() is not None:
            raise ValueError(f"ZIP CRC failed: {path}")
        for name, digest in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Archived mask hash differs: {name}")
        for name, content in metadata.items():
            if archive.read(name) != content:
                raise ValueError(f"Archived metadata differs: {name}")


def write_or_verify(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"Symlink at delivery path: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"Existing delivery metadata differs: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def package_cell(source: dict, data_root: Path, output_root: Path) -> dict:
    cell = source["cell"]
    records = source["records"]
    folder = output_root / cell
    if folder.is_symlink():
        raise ValueError(f"Symlink at delivery path: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    archive_path = folder / ARCHIVE_NAME
    digest_path = folder / f"{ARCHIVE_NAME}.sha256"
    portable = {
        "schema_version": 1,
        "cell": cell,
        "input_folder": f"organoid_data/{cell}/fluo",
        "output_folder": f"organoid_data/{cell}/{PREFIX}",
        "method": METHOD,
        "scientific_status": SCIENTIFIC_STATUS,
        "purpose": "automated LD predictions for expert review; not manual ground truth",
        "independent_external_accuracy_known": False,
        "mask_count": len(records),
        "mask_dtype": "uint8",
        "background_value": 0,
        "foreground_value": 255,
        "photometric": "MINISBLACK",
        "normalization_percentiles": [1.0, 99.8],
        "patch_size": 512,
        "overlap": 128,
        "votes_required": 1,
        "folds": source["folds"],
        "source_prediction_manifest_sha256": source["source_manifest_sha256"],
        "files": records,
    }
    note = (
        "These TIFFs are exploratory lipid-droplet predictions selected for expert "
        "review, not manually labelled ground truth. A pixel is foreground when "
        "at least one of five locked focal folds votes positive. This rule has "
        "no independent external accuracy estimate. Each single-channel TIFF "
        "uses the original input grid, uint8, MINISBLACK, background=0 and "
        "foreground=255. The archive contains no source images, checkpoints, "
        "full local manifests, logs, credentials or absolute computer paths.\n"
    ).encode("utf-8")
    hashes = "".join(
        f"{r['mask_sha256']}  {PREFIX}/{r['filename']}\n" for r in records
    ).encode("ascii")
    metadata = {
        f"{PREFIX}/PORTABLE_MANIFEST.json": canonical_json(portable),
        f"{PREFIX}/AUTO_LABELS_README.txt": note,
        f"{PREFIX}/SHA256SUMS.txt": hashes,
    }
    if archive_path.is_symlink() or digest_path.is_symlink():
        raise ValueError(f"Symlink at delivery path: {cell}")
    if archive_path.exists() or digest_path.exists():
        if not archive_path.is_file() or not digest_path.is_file():
            raise ValueError(f"Incomplete existing package: {cell}")
        digest_line = digest_path.read_text(encoding="ascii").strip().split()
        if (len(digest_line) != 2 or digest_line[1] != ARCHIVE_NAME
                or digest_line[0] != sha256_file(archive_path)):
            raise ValueError(f"Existing ZIP checksum differs: {cell}")
        verify_archive(archive_path, records, metadata)
    else:
        partial = archive_path.with_suffix(".zip.partial")
        if partial.exists():
            raise FileExistsError(f"Retained incomplete ZIP needs inspection: {partial}")
        with zipfile.ZipFile(partial, "x", allowZip64=True) as archive:
            for record in records:
                name = record["filename"]
                zip_entry(archive, f"{PREFIX}/{name}",
                          (data_root / cell / PREFIX / name).read_bytes())
            for name, content in metadata.items():
                zip_entry(archive, name, content)
        verify_archive(partial, records, metadata)
        if partial.stat().st_size >= MAX_GIT_BLOB_BYTES:
            raise ValueError(f"ZIP exceeds GitHub single-file limit: {cell}")
        os.rename(partial, archive_path)
        write_or_verify(digest_path,
                        f"{sha256_file(archive_path)}  {ARCHIVE_NAME}\n".encode("ascii"))
    return {
        "cell": cell,
        "mask_count": len(records),
        "archive": f"{cell}/{ARCHIVE_NAME}",
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "source_prediction_manifest_sha256": source["source_manifest_sha256"],
    }


def write_index(output_root: Path, results: list[dict], folds: list[dict]) -> None:
    total = sum(item["mask_count"] for item in results)
    if len(results) != 10 or total != 1330:
        raise ValueError("Expected ten cells and 1,330 masks")
    index = {
        "schema_version": 1,
        "method": METHOD,
        "scientific_status": SCIENTIFIC_STATUS,
        "votes_required": 1,
        "cell_count": len(results),
        "mask_count": total,
        "folds": folds,
        "packages": results,
    }
    checksums = "".join(
        f"{item['archive_sha256']}  {item['archive']}\n" for item in results
    )
    rows = "\n".join(
        f"| `{item['cell']}` | {item['mask_count']} | {item['archive_bytes']:,} |"
        for item in results
    )
    readme = f"""# LD 1/5 自动预测 mask 交付包

本目录是 **10 个细胞、1,330 张**脂滴自动预测 mask 的独立 1/5 投票交付件。每个细胞的 `{ARCHIVE_NAME}` 解压为 `{PREFIX}/`，TIFF 与该细胞 `fluo/` 输入同名、尺寸与像素坐标一致，均为单通道 `uint8`、`MINISBLACK`、背景 0、前景 255。每包内的 `PORTABLE_MANIFEST.json` 记录输入和 mask SHA256、模型哈希与阈值；`SHA256SUMS.txt` 可核对解压后的 mask。相邻 `.sha256` 文件及本目录 `SHA256SUMS.txt` 可核对完整 ZIP；`PACKAGE_INDEX.json` 记录十包的数量、大小和哈希。

| 细胞 | mask 数量 | ZIP 字节数 |
| --- | ---: | ---: |
{rows}
| **合计** | **{total}** | **{sum(item['archive_bytes'] for item in results):,}** |

五个 focal U-Net 折模型使用锁定的验证阈值 `[0.65, 0.80, 0.60, 0.35, 0.50]`。逐图 p1–p99.8 归一化，512×512 滑窗、重叠 128 像素。**至少一折**投票为前景时输出 LD。这是负责人目视偏好的探索性人工复核版本，**不是人工 ground truth**，投票规则没有独立外部准确率估计。它独立于此前的 3/5 交付包，解压时也使用独立的 `{PREFIX}/` 文件夹。

ZIP 只含生成的 mask 和脱敏元数据；不含原始荧光图像、人工标签、模型权重、完整本地预测 manifest、日志、凭据或本机绝对路径。请勿将其当作 `label_ref` 的人工训练标签。由[独立打包脚本](../../code/package_ld_vote1_auto_labels.py)从本地已核验输出生成。

下载后在本目录运行 `env LC_ALL=C shasum -a 256 -c SHA256SUMS.txt` 验证十个 ZIP。若只验证单包，请进入该细胞的子目录，再运行 `env LC_ALL=C shasum -a 256 -c {ARCHIVE_NAME}.sha256`。把 ZIP 解压到新的空目录，避免覆盖已有预测文件；ZIP 内的 `{PREFIX}/SHA256SUMS.txt` 可继续核对逐张 mask。[旧版 3/5 结果](../ld_auto_labels/README.md)和[生成与验证报告](../../reports/ld_vote1_auto_labels_20261015/README.md)保留完整方法边界。
"""
    write_or_verify(output_root / "PACKAGE_INDEX.json", canonical_json(index))
    write_or_verify(output_root / "SHA256SUMS.txt", checksums.encode("ascii"))
    write_or_verify(output_root / "README.md", readme.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "Desktop/Organoid/organoid_data")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "deliverables/ld_vote1_auto_labels")
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    requested_output_root = args.output_root.expanduser()
    if requested_output_root.is_symlink():
        raise ValueError("Delivery root must not be a symlink")
    output_root = requested_output_root.resolve()
    if (not data_root.is_dir() or output_root.name != "ld_vote1_auto_labels"
            or output_root.is_relative_to(data_root)
            or "ld_auto_labels" in output_root.parts):
        raise ValueError("Use the existing source root and a separate ld_vote1_auto_labels delivery")
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir()):
        raise ValueError("Delivery root must be a regular directory")
    inference_code = Path(__file__).with_name("ld_focal_vote1_predict.py")
    inference_code_hash = sha256_file(inference_code)

    # No delivery is written until all ten sources pass the full preflight.
    sources = [verify_source(cell, data_root, inference_code_hash) for cell in CELL_COUNTS]
    if sum(len(source["records"]) for source in sources) != 1330:
        raise ValueError("Expected exactly 1,330 verified masks")
    output_root.mkdir(parents=True, exist_ok=True)
    results = [package_cell(source, data_root, output_root) for source in sources]
    write_index(output_root, results, sources[0]["folds"])
    print(json.dumps({"status": "PASS", "cell_count": 10,
                      "mask_count": 1330, "packages": results}, indent=2))


if __name__ == "__main__":
    main()
