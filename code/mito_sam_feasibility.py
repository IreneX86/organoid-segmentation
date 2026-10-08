"""CPU-only SAM feasibility; prediction and GT scoring are separate commands.

Original TIFFs are read-only. All binary outputs retain native dimensions.
This pilot uses published OOF strata for coverage, not candidate selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def select_cases(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for low in (True, False):
        group = [r for r in rows if (float(r["foreground_fraction_true"]) <= 0.05) == low]
        median = float(np.median([float(r["dice"]) for r in group]))
        for weak in (True, False):
            cell = [r for r in group if (float(r["dice"]) < median) == weak]
            row = min(
                cell,
                key=lambda r: hashlib.sha256(
                    ("20261008-sam|" + r["filename"]).encode()
                ).hexdigest(),
            )
            selected.append(
                {
                    "filename": row["filename"],
                    "fold": int(row["fold"]),
                    "stratum": (
                        f"{'low' if low else 'high'}_foreground_"
                        f"{'low' if weak else 'high'}_baseline_dice"
                    ),
                }
            )
    return selected


def normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99.8])
    return np.clip((image.astype(np.float32) - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(
        np.float32
    )


def origins(size: int, patch: int = 512, stride: int = 384) -> list[int]:
    if size < patch:
        raise ValueError("This preregistered pilot requires dimensions >=512.")
    return sorted(set(list(range(0, size - patch + 1, stride)) + [size - patch]))


def boxes_from_proposal(proposal: np.ndarray) -> np.ndarray:
    """Input/proposal only. Connected regions are NOT biological instances."""
    labels, _ = ndi.label(proposal, structure=np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    boxes = []
    for label_id, slices in enumerate(ndi.find_objects(labels), 1):
        if slices is None or counts[label_id] < 10:
            continue
        ys, xs = slices
        boxes.append(
            [
                max(0, xs.start - 4),
                max(0, ys.start - 4),
                min(proposal.shape[1], xs.stop + 4),
                min(proposal.shape[0], ys.stop + 4),
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def load_predictor(name: str, models: Path):
    import torch

    if name == "microsam_lm":
        from segment_anything import SamPredictor, sam_model_registry

        path = models / "microsam_vit_b_lm.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model_state" in state:
            state = state["model_state"]
        state = {k.removeprefix("sam."): v for k, v in state.items()}
        model = sam_model_registry["vit_b"]()
        model.load_state_dict(state, strict=True)
        model.eval().to("cpu")
        return SamPredictor(model), path
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    path = models / "sam2.1_hiera_tiny.pt"
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(path), device="cpu")
    model.eval()
    # CUDA connected-components postprocessing is not used in this CPU pilot.
    return SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0), path


def decode(predictor, name: str, boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import torch

    prob = np.zeros(shape, np.float32)
    for start in range(0, len(boxes), 8):
        batch = boxes[start : start + 8]
        if name == "microsam_lm":
            transformed = predictor.transform.apply_boxes_torch(torch.as_tensor(batch), shape)
            logits, _, _ = predictor.predict_torch(
                None, None, boxes=transformed, multimask_output=False, return_logits=True
            )
            values = torch.sigmoid(logits[:, 0]).numpy()
        else:
            logits, _, _ = predictor.predict(box=batch, multimask_output=False, return_logits=True)
            logits = np.asarray(logits).reshape(-1, *shape)
            values = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        prob = np.maximum(prob, values.max(axis=0))
    return prob


def predict(args) -> None:
    import psutil
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PREDICTIONS_COMPLETE.json").exists():
        raise FileExistsError("Completed pilot exists; do not overwrite or retune.")
    manifest_path = output / "manifest.json"
    cases = select_cases(args.oof_csv)
    manifest = {
        "protocol": "experiments/mito_sam_feasibility_protocol.md",
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "ram_gib": psutil.virtual_memory().total / 2**30,
        "cases": cases,
        "prompt_source": "image_otsu OR frozen baseline OOF prediction; never GT",
        "models": {},
        "timing": [],
    }
    for case in cases:
        case["input_sha256"] = digest(args.image_dir / case["filename"])
        proposal = (
            args.baseline_root
            / f"fold_{case['fold']}"
            / "predictions"
            / "validation"
            / case["filename"]
        )
        case["baseline_prediction_sha256"] = digest(proposal)
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["cases"] != cases:
            raise ValueError("Resume inputs/proposals differ from the locked manifest.")
        manifest = previous
    elif manifest_path.exists():
        raise FileExistsError(
            "Partial pilot exists. Use --resume only for a technical interruption."
        )
    write_json(manifest_path, manifest)
    for name in ("sam2_tiny", "microsam_lm"):
        completed = {row["filename"] for row in manifest["timing"] if row["model"] == name}
        if len(completed) == len(cases):
            continue
        start_load = time.perf_counter()
        predictor, checkpoint = load_predictor(name, args.models)
        manifest["models"][name] = {
            "sha256": digest(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "load_seconds": time.perf_counter() - start_load,
        }
        with torch.inference_mode():
            for case in cases:
                filename = case["filename"]
                if filename in completed:
                    # Reuse finished predictions; never rerun based on quality.
                    for source in ("otsu", "baseline"):
                        stored = tifffile.imread(output / f"{name}_{source}" / filename)
                        if stored.shape != (1200, 1200) or not np.isin(stored, [0, 1]).all():
                            raise ValueError("Completed prediction failed integrity checks.")
                    continue
                already = [
                    output / method / filename for method in (f"{name}_otsu", f"{name}_baseline")
                ]
                if all(p.exists() for p in already):
                    raise FileExistsError(
                        "Partial rerun requires an explicitly new output directory."
                    )
                begin = time.perf_counter()
                image = normalize(tifffile.imread(args.image_dir / filename))
                smooth = ndi.gaussian_filter(image, 1)
                otsu = smooth > threshold_otsu(smooth)
                baseline = (
                    tifffile.imread(
                        args.baseline_root
                        / f"fold_{case['fold']}"
                        / "predictions"
                        / "validation"
                        / filename
                    )
                    > 0
                )
                assert baseline.shape == image.shape == (1200, 1200)
                proposals = {"otsu": otsu, "baseline": baseline}
                sums = {key: np.zeros(image.shape, np.float32) for key in proposals}
                count = np.zeros(image.shape, np.float32)
                prompts = {key: [] for key in proposals}
                encode_seconds = 0.0
                decode_seconds = {key: 0.0 for key in proposals}
                for y in origins(image.shape[0]):
                    for x in origins(image.shape[1]):
                        tile = image[y : y + 512, x : x + 512]
                        rgb = np.repeat(
                            np.round(tile[..., None] * 255).astype(np.uint8), 3, axis=-1
                        )
                        tick = time.perf_counter()
                        predictor.set_image(rgb)
                        encode_seconds += time.perf_counter() - tick
                        for source, proposal in proposals.items():
                            boxes = boxes_from_proposal(proposal[y : y + 512, x : x + 512])
                            prompts[source].append({"x": x, "y": y, "boxes": boxes.tolist()})
                            tick = time.perf_counter()
                            prob = decode(predictor, name, boxes, tile.shape)
                            decode_seconds[source] += time.perf_counter() - tick
                            sums[source][y : y + 512, x : x + 512] += prob
                        count[y : y + 512, x : x + 512] += 1
                        print(
                            json.dumps(
                                {
                                    "model": name,
                                    "image": filename,
                                    "tile": [y, x],
                                    "elapsed_s": round(time.perf_counter() - begin, 1),
                                }
                            ),
                            flush=True,
                        )
                assert count.min() >= 1
                for source in proposals:
                    method = f"{name}_{source}"
                    folder = output / method
                    folder.mkdir(exist_ok=True)
                    prob = sums[source] / count
                    tifffile.imwrite(folder / filename, (prob >= 0.5).astype(np.uint8))
                    tifffile.imwrite(folder / (Path(filename).stem + "_probability.tif"), prob)
                    write_json(folder / (Path(filename).stem + "_prompts.json"), prompts[source])
                control = output / "otsu_control"
                control.mkdir(exist_ok=True)
                tifffile.imwrite(control / filename, otsu.astype(np.uint8))
                manifest["timing"].append(
                    {
                        "model": name,
                        "filename": filename,
                        "encode_seconds": encode_seconds,
                        "decode_seconds": decode_seconds,
                        "total_seconds": time.perf_counter() - begin,
                        "prompt_counts": {
                            key: sum(len(t["boxes"]) for t in val) for key, val in prompts.items()
                        },
                        "process_peak_rss_bytes": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    }
                )
                write_json(manifest_path, manifest)
        del predictor
    locked = {
        str(p.relative_to(output)): digest(p)
        for p in output.rglob("*")
        if p.is_file() and p.suffix in (".tif", ".json")
    }
    write_json(output / "PREDICTIONS_COMPLETE.json", locked)
    print("PREDICTIONS_COMPLETE — GT has not been read", flush=True)


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    pb = pred & ~ndi.binary_erosion(pred)
    gb = truth & ~ndi.binary_erosion(truth)
    bp = float(np.mean(ndi.distance_transform_edt(~gb)[pb] <= 2)) if pb.any() else 0.0
    br = float(np.mean(ndi.distance_transform_edt(~pb)[gb] <= 2)) if gb.any() and pb.any() else 0.0
    return {
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "distant_fp": int(
            np.count_nonzero(pred & ~truth & (ndi.distance_transform_edt(~truth) > 10))
        ),
        "boundary_f1_2px": 2 * bp * br / max(bp + br, 1e-12),
        "foreground_fraction": float(truth.mean()),
    }


def evaluate(args) -> None:
    output = args.output
    locked = json.loads((output / "PREDICTIONS_COMPLETE.json").read_text())
    assert all(digest(output / path) == value for path, value in locked.items()), (
        "Prediction lock mismatch"
    )
    manifest = json.loads((output / "manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        filename = case["filename"]
        truth = tifffile.imread(args.mask_dir / filename) > 0
        paths = {
            "baseline": args.baseline_root
            / f"fold_{case['fold']}"
            / "predictions"
            / "validation"
            / filename
        }
        paths.update(
            {
                method: output / method / filename
                for method in (
                    "otsu_control",
                    "microsam_lm_otsu",
                    "microsam_lm_baseline",
                    "sam2_tiny_otsu",
                    "sam2_tiny_baseline",
                )
            }
        )
        for method, path in paths.items():
            prediction = tifffile.imread(path) > 0
            assert prediction.shape == truth.shape == (1200, 1200)
            rows.append(
                {
                    "method": method,
                    "filename": filename,
                    "stratum": case["stratum"],
                    **metrics(prediction, truth),
                }
            )
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        summary[method] = {
            key: float(np.mean([r[key] for r in group]))
            for key in (
                "dice",
                "iou",
                "precision",
                "recall",
                "fp",
                "fn",
                "distant_fp",
                "boundary_f1_2px",
            )
        }
        summary[method]["low_foreground_dice"] = float(
            np.mean([r["dice"] for r in group if r["foreground_fraction"] <= 0.05])
        )
        summary[method]["low_foreground_precision"] = float(
            np.mean([r["precision"] for r in group if r["foreground_fraction"] <= 0.05])
        )
        summary[method]["min_dice"] = min(r["dice"] for r in group)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("predict")
    infer.add_argument("--image-dir", type=Path, required=True)
    infer.add_argument("--oof-csv", type=Path, required=True)
    infer.add_argument("--models", type=Path, required=True)
    infer.add_argument(
        "--resume",
        action="store_true",
        help="Reuse finished cases after technical interruption only.",
    )
    for task in (infer, sub.add_parser("evaluate")):
        task.add_argument("--output", type=Path, required=True)
        task.add_argument("--baseline-root", type=Path, required=True)
        if task is not infer:
            task.add_argument("--mask-dir", type=Path, required=True)
    args = parser.parse_args()
    (predict if args.command == "predict" else evaluate)(args)


if __name__ == "__main__":
    main()
