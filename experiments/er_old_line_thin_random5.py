"""Five reproducibly sampled ER planes: accepted line radius 4 vs opt-in 3.

This reads original ER/Fluo stacks and their frozen base masks. It leaves all
historical TIFFs unchanged and does not use the separate mesh-line candidate.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from er_bright_line_pipeline import narrow_existing_lines, process_slice, settings_for_profile, sha256  # noqa: E402

PLAN = ROOT / "outputs/er_random5_profiled_20261028/sampling_plan.json"
DEST = ROOT / "outputs/er_old_line_thin_random5_20261028"
SEED = 20261028


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def display(er: np.ndarray, mask: np.ndarray | None = None, *, difference: bool = False) -> Image.Image:
    low, high = np.percentile(er, (1, 99.8))
    gray = np.clip((er.astype(np.float32)-low)/max(high-low, 1)*255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if mask is not None:
        selected = mask > 0
        color = np.array([255, 70, 70] if difference else [255, 205, 0])
        rgb[selected] = (0.45 * rgb[selected] + 0.55 * color).astype(np.uint8)
    return Image.fromarray(rgb).resize((600, 600), Image.Resampling.LANCZOS)


def render_overview(records: list[dict]) -> Path:
    """Compact index; per-cell TIFFs and comparison PNGs retain full detail."""
    width, height = 3 * 360, len(records) * 390
    sheet = Image.new("RGB", (width, height), "#101010")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        comparison = Image.open(record["comparison"])
        draw.text((8, row * 390 + 4),
                  f"{record['cell']}  Z{record['source_z_one_based']}  |  original / old / thinner",
                  fill="white")
        for col in range(3):
            panel = comparison.crop((col * 600, 36, (col + 1) * 600, 636))
            sheet.paste(panel.resize((360, 360), Image.Resampling.LANCZOS),
                        (col * 360, row * 390 + 30))
    path = DEST / "random5_overview.png"
    sheet.save(path)
    return path


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Refusing overwrite: {DEST}")
    plan = json.loads(PLAN.read_text())
    rng = np.random.default_rng(SEED)
    spheroids = [row for row in plan["cells"] if row["profile"] == "spheroid_default"]
    dim = [row for row in plan["cells"] if row["profile"] == "dim_organoid_candidate"]
    if len(spheroids) != 7 or len(dim) != 1:
        raise ValueError("Unexpected source plan; do not silently change sample design")
    selection = [spheroids[int(i)] for i in rng.choice(len(spheroids), size=4, replace=False)] + dim
    records = []
    DEST.mkdir()
    for row in selection:
        er_path, fluo_path, base_path = (Path(row[key]) for key in ("source_er", "source_fluo", "base_mask"))
        with (tifffile.TiffFile(er_path) as er_file,
              tifffile.TiffFile(fluo_path) as fluo_file,
              tifffile.TiffFile(base_path) as base_file):
            shape = er_file.series[0].shape
            if shape != fluo_file.series[0].shape or shape != base_file.series[0].shape:
                raise ValueError(f"Misaligned inputs for {row['cell']}")
            depth = shape[0]
            # Uniform random central-80% plane, excluding the 5 previously
            # reviewed planes. One dim organoid is deliberately included.
            candidates = np.array([z for z in range(depth // 10, depth - depth // 10)
                                   if z not in row["z_zero_based"]])
            z = int(rng.choice(candidates))
            er = er_file.pages[z].asarray()
            fluo = fluo_file.pages[z].asarray()
            base = base_file.pages[z].asarray()
        base_manifest_path = base_path.with_name("ER_mask_manifest.json")
        bbox = None
        if base_manifest_path.is_file():
            base_manifest = json.loads(base_manifest_path.read_text())
            for entry in base_manifest.get("slices", []):
                if entry.get("source_z_zero_based") == z:
                    bbox = entry.get("strong", {}).get("artifact_group_bbox_yxyx")
                    break
        settings = settings_for_profile(row["profile"])
        bright, old, _ = process_slice(er, fluo, base, settings=settings,
                                       artifact_bbox_yxyx=bbox)
        thin_bool, restored = narrow_existing_lines(bright > 0, old > 0, half_width=3)
        new = thin_bool.astype(np.uint8) * 255
        if not (er.shape == fluo.shape == base.shape == old.shape == new.shape):
            raise RuntimeError("Output shape mismatch")
        if not np.all(new <= bright):
            raise RuntimeError("Thinner line includes pixels outside fixed bright candidate")
        if not np.all(new <= old):
            raise RuntimeError("Thinner line must not add pixels relative to old line")
        folder = DEST / row["cell"]
        folder.mkdir()
        paths = {"er": folder / "ER_original.tif", "fluo": folder / "Fluo_original.tif",
                 "old": folder / "ER_mask_line_radius4.tif",
                 "new": folder / "ER_mask_line_radius3.tif"}
        for key, array in (("er", er), ("fluo", fluo), ("old", old), ("new", new)):
            tifffile.imwrite(paths[key], array, compression="zlib")
        board = Image.new("RGB", (2400, 636), "#101010")
        draw = ImageDraw.Draw(board)
        parts = (display(er), display(er, old), display(er, new), display(er, (old > 0) & (new == 0), difference=True))
        names = ("original ER", "old line radius 4", "slightly thinner radius 3", "removed pixels in red")
        for i, (part, name) in enumerate(zip(parts, names)):
            draw.text((i * 600 + 8, 8), f"{row['cell']} Z{z + 1} {name}", fill="white")
            board.paste(part, (i * 600, 36))
        comparison = folder / "comparison.png"
        board.save(comparison)
        record = {"cell": row["cell"], "profile": row["profile"], "source_z_zero_based": z,
                  "source_z_one_based": z + 1, "source_er": str(er_path), "source_fluo": str(fluo_path),
                  "base_mask": str(base_path), "source_plane_sha256": {"er": sha_array(er),
                  "fluo": sha_array(fluo), "base": sha_array(base)},
                  "artifact_bbox_yxyx": bbox,
                  "old_line_pixels": int(np.count_nonzero(old)), "thin_line_pixels": int(np.count_nonzero(new)),
                  "removed_pixels": int(np.count_nonzero((old > 0) & (new == 0))),
                  "added_pixels": int(np.count_nonzero((old == 0) & (new > 0))),
                  "restored_components": int(restored), "comparison": str(comparison),
                  "outputs": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}}
        records.append(record)
        print(f"{row['cell']} Z{z + 1}: {record['old_line_pixels']} -> {record['thin_line_pixels']}", flush=True)
    (DEST / "manifest.json").write_text(json.dumps({"status": "visual candidate; no original changed",
        "seed": SEED, "selection": "4 spheroid cells + 1 dim organoid; one random central-80% plane per cell; exclude prior 5 planes",
        "old_line_half_width": 4, "new_line_half_width": 3, "records": records}, indent=2) + "\n")
    render_overview(records)


if __name__ == "__main__":
    main()
