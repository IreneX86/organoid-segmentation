#!/usr/bin/env python3
"""Retrospective, center-only audit of the frozen Nuc review markers.

Uses the existing image heuristic without changing its parameters or ranking.
The 34-image same-batch validation set is not an independent test, and the
``nuc_###`` numbering is not interpreted as an acquisition or Z sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from nuc_cross_slice_review import (
    EDGE_MARGIN_PX,
    LARGE_BLUR_PX,
    MAX_MARKERS_PER_SLICE,
    PEAK_SEPARATION_PX,
    SMALL_BLUR_PX,
    candidate_rows,
    contrast_bounds,
    dark_core_response,
)


MIN_AREA = 100  # Same GT component definition as the prior object audit.
MISS_COVERAGE = 0.10
TOLERANCE_PX = 10  # Sensitivity check; exact center-in-GT is primary.
RANK_CUTOFFS = (10, 20, 40)
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_tiff(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = tifffile.imread(path)
    if image.shape != shape or image.ndim != 2:
        raise ValueError(f"Wrong mask shape: {path.name}")
    values = np.unique(image)
    if not set(values.tolist()).issubset({0, 1, 255}):
        raise ValueError(f"Mask must be binary: {path.name}")
    return image != 0


def edges_for_markers(
    rows: list[dict[str, object]], labels: np.ndarray, valid_ids: set[int],
    tolerance: int,
) -> list[list[tuple[int, int]]]:
    """For each marker, collect eligible GT components within a pixel radius."""
    height, width = labels.shape
    edges: list[list[tuple[int, int]]] = []
    for row in rows:
        x, y = int(row["x"]), int(row["y"])
        x0, x1 = max(0, x - tolerance), min(width, x + tolerance + 1)
        y0, y1 = max(0, y - tolerance), min(height, y + tolerance + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance_squared = (xx - x) ** 2 + (yy - y) ** 2
        nearby = labels[y0:y1, x0:x1]
        keep = (distance_squared <= tolerance ** 2) & (nearby != 0)
        near_ids = np.unique(nearby[keep])
        choices = []
        for object_id in near_ids:
            object_id = int(object_id)
            if object_id in valid_ids:
                shortest = int(distance_squared[(nearby == object_id) & keep].min())
                choices.append((object_id, shortest))
        edges.append(sorted(choices, key=lambda pair: (pair[1], pair[0])))
    return edges


def one_to_one_matches(
    edges: list[list[tuple[int, int]]], marker_indices: list[int],
) -> dict[int, int]:
    """Maximum cardinality marker-to-GT matching; original rank breaks ties."""
    owner: dict[int, int] = {}

    def augment(marker: int, seen: set[int]) -> bool:
        for object_id, _ in edges[marker]:
            if object_id in seen:
                continue
            seen.add(object_id)
            previous = owner.get(object_id)
            if previous is None or augment(previous, seen):
                owner[object_id] = marker
                return True
        return False

    for marker in marker_indices:
        augment(marker, set())
    return owner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    root = args.project_root
    output_json = args.output_json or root / "reports/nuc_dark_core_validation_audit_20261017.json"
    output_md = args.output_md or root / "reports/nuc_dark_core_validation_audit_20261017.md"
    manifest_path = root / "experiments/results/nuc_internal_split_v1.json"
    model_dir = root / "runpod_results/nuc_internal_20261015/full_results/model"
    run_manifest = json.loads((model_dir / "run_manifest.json").read_text())
    threshold_path = model_dir / "threshold.json"
    if (run_manifest.get("target") != "nuc"
            or run_manifest.get("purpose") != "same_batch_internal_diagnostic"
            or sha256(manifest_path) != run_manifest.get("split_sha256")
            or sha256(model_dir / "split.json") != run_manifest.get("split_sha256")
            or sha256(threshold_path) != run_manifest.get("threshold_sha256")):
        raise ValueError("Frozen split or threshold hash differs from run manifest")
    manifest = json.loads(manifest_path.read_text())
    names = manifest["validation"]
    if len(names) != 34 or len(set(names)) != 34:
        raise ValueError("Expected the frozen 34-image validation split")
    threshold = float(json.loads(threshold_path.read_text())["threshold"])
    if threshold != 0.45:
        raise ValueError("Frozen threshold changed")
    prediction_dir = model_dir / "validation_artifacts/predictions/selected"
    probability_dir = model_dir / "validation_artifacts/probabilities"

    per_image = []
    object_details = []
    totals = {str(k): {"markers": 0, "outside_prediction_queue": 0,
                       "centers_inside_gt": 0, "queue_centers_inside_gt": 0,
                       "exact_gt_hits": 0, "exact_missed_hits": 0,
                       "within_10px_gt_hits": 0, "within_10px_missed_hits": 0,
                       "queue_exact_gt_hits": 0, "queue_exact_missed_hits": 0,
                       "queue_within_10px_gt_hits": 0,
                       "queue_within_10px_missed_hits": 0} for k in RANK_CUTOFFS}
    gt_total = missed_total = edge_inaccessible_missed = edge_touched_missed = 0
    for name in names:
        input_path = args.input_dir / name
        label_path = args.label_dir / name
        if sha256(input_path) != manifest["input_sha256"][name]:
            raise ValueError(f"Input hash mismatch: {name}")
        if sha256(label_path) != manifest["label_sha256"][name]:
            raise ValueError(f"GT hash mismatch: {name}")
        image = tifffile.imread(input_path)
        if image.ndim != 2 or not np.issubdtype(image.dtype, np.number):
            raise ValueError(f"Not a 2D numeric image: {name}")
        gt = binary_tiff(label_path, image.shape)
        if name == "nuc_005.tif" and np.any(gt):
            raise ValueError("Archived nuc_005 GT is not fully black")
        pred = binary_tiff(prediction_dir / name, image.shape)
        probability = tifffile.imread(probability_dir / name)
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise ValueError(f"Invalid archived probability: {name}")
        if not np.array_equal(pred, probability >= threshold):
            raise ValueError(f"Selected mask differs from archived probability: {name}")
        lo, hi = contrast_bounds(image)
        response = dark_core_response(image, lo, hi)
        rows = candidate_rows(response, None, None, pred, name)
        if len(rows) > MAX_MARKERS_PER_SLICE or any(row["available_neighbors"] for row in rows):
            raise ValueError(f"Invalid center-only marker list: {name}")

        labels, count = ndimage.label(gt, structure=CONNECTIVITY_8)
        area = np.bincount(labels.ravel(), minlength=count + 1)
        covered = np.bincount(labels[pred], minlength=count + 1)
        margin = min(EDGE_MARGIN_PX, (image.shape[0] - 1) // 2,
                     (image.shape[1] - 1) // 2)
        inside = labels[margin:image.shape[0] - margin,
                        margin:image.shape[1] - margin]
        accessible_pixels = np.bincount(inside.ravel(), minlength=count + 1)
        slices = ndimage.find_objects(labels)
        objects: dict[int, dict[str, object]] = {}
        for object_id in range(1, count + 1):
            if area[object_id] < MIN_AREA:
                continue
            bounds = slices[object_id - 1]
            assert bounds is not None
            y0, y1 = bounds[0].start, bounds[0].stop
            x0, x1 = bounds[1].start, bounds[1].stop
            coverage = float(covered[object_id] / area[object_id])
            touches_margin = x0 < margin or y0 < margin or x1 > image.shape[1] - margin or y1 > image.shape[0] - margin
            accessible = bool(accessible_pixels[object_id])
            objects[object_id] = {"coverage": coverage, "area_px": int(area[object_id]),
                                  "bbox_xyxy": [x0, y0, x1, y1],
                                  "touches_48px_margin": touches_margin,
                                  "has_gt_pixel_in_allowed_center_area": accessible}
            if coverage < MISS_COVERAGE:
                edge_touched_missed += int(touches_margin)
                edge_inaccessible_missed += int(not accessible)
        valid_ids = set(objects)
        missed_ids = {k for k in valid_ids if float(objects[k]["coverage"]) < MISS_COVERAGE}
        gt_total += len(valid_ids)
        missed_total += len(missed_ids)
        exact_edges = edges_for_markers(rows, labels, valid_ids, 0)
        near_edges = edges_for_markers(rows, labels, valid_ids, TOLERANCE_PX)
        for cutoff in RANK_CUTOFFS:
            indices = [i for i, row in enumerate(rows) if int(row["rank"]) <= cutoff]
            queue = [i for i in indices if not bool(rows[i]["inside_frozen_mask_at_point"])]
            bucket = totals[str(cutoff)]
            bucket["markers"] += len(indices)
            bucket["outside_prediction_queue"] += len(queue)
            bucket["centers_inside_gt"] += sum(bool(gt[int(rows[i]["y"]), int(rows[i]["x"])]) for i in indices)
            bucket["queue_centers_inside_gt"] += sum(bool(gt[int(rows[i]["y"]), int(rows[i]["x"])]) for i in queue)
            for prefix, chosen in (("", indices), ("queue_", queue)):
                for label, edges in (("exact_", exact_edges), ("within_10px_", near_edges)):
                    matches = one_to_one_matches(edges, chosen)
                    bucket[f"{prefix}{label}gt_hits"] += len(matches)
                    bucket[f"{prefix}{label}missed_hits"] += len(set(matches) & missed_ids)
        for object_id in sorted(missed_ids):
            item = objects[object_id]
            item_out = {"image": name, "area_px": item["area_px"],
                        "bbox_xyxy": item["bbox_xyxy"],
                        "baseline_coverage": round(float(item["coverage"]), 6),
                        "touches_48px_margin": item["touches_48px_margin"],
                        "has_gt_pixel_in_allowed_center_area": item["has_gt_pixel_in_allowed_center_area"]}
            for field, edges in (("first_exact_rank", exact_edges), ("first_within_10px_rank", near_edges)):
                ranks = [int(rows[i]["rank"]) for i, choices in enumerate(edges)
                         if object_id in {k for k, _ in choices}]
                item_out[field] = min(ranks) if ranks else None
            object_details.append(item_out)
        per_image.append({"image": name, "gt_components_ge_100px": len(valid_ids),
                          "missed_gt_components": len(missed_ids),
                          "markers_top40": len(rows),
                          "outside_prediction_queue_top40": sum(not bool(row["inside_frozen_mask_at_point"]) for row in rows),
                          "centers_inside_gt_top40": sum(bool(gt[int(row["y"]), int(row["x"])]) for row in rows),
                          "baseline_predicted_positive_pixels": int(pred.sum())})
        print(f"{name}: {len(rows)} markers; {len(missed_ids)} missed GT", flush=True)

    if gt_total != 1175 or missed_total != 22:
        raise ValueError(f"GT component counts differ from prior audit: {gt_total}, {missed_total}")
    for bucket in totals.values():
        bucket["centers_outside_gt"] = bucket["markers"] - bucket["centers_inside_gt"]
        bucket["queue_centers_outside_gt"] = (
            bucket["outside_prediction_queue"] - bucket["queue_centers_inside_gt"]
        )
    negative = next(item for item in per_image if item["image"] == "nuc_005.tif")
    if negative["gt_components_ge_100px"] or negative["centers_inside_gt_top40"]:
        raise ValueError("Archived nuc_005 must be GT-negative")
    negative["markers_outside_gt_top40"] = negative["markers_top40"]
    negative["queue_markers_outside_gt_top40"] = negative["outside_prediction_queue_top40"]
    missed_images = len({item["image"] for item in object_details})
    hit_missed_images = len({item["image"] for item in object_details
                             if item["first_exact_rank"] is not None})
    payload = {
        "purpose": "retrospective same-batch development diagnostic; no independent source validation",
        "source": {"split": "experiments/results/nuc_internal_split_v1.json",
                   "heuristic": "code/nuc_cross_slice_review.py",
                   "prediction": "archived selected threshold 0.45",
                   "source_input_label_sha256": "all 34 match frozen split",
                   "split_threshold_sha256": "match archived run_manifest.json",
                   "selected_prediction_vs_probability": "all 34 exact at >=0.45"},
        "settings": {"image_only_center_response": True, "adjacent_nuc_indices_used": False,
                     "small_blur_px": SMALL_BLUR_PX, "large_blur_px": LARGE_BLUR_PX,
                     "peak_separation_px": PEAK_SEPARATION_PX,
                     "edge_margin_px": EDGE_MARGIN_PX,
                     "max_markers_per_image": MAX_MARKERS_PER_SLICE,
                     "gt_connectivity": 8, "minimum_gt_area_px": MIN_AREA,
                     "missed_baseline_coverage_lt": MISS_COVERAGE,
                     "sensitivity_distance_px": TOLERANCE_PX,
                     "queue_policy": "filter frozen-mask-inside markers after original ranking, no refill"},
        "denominators": {"validation_images": len(names), "nonempty_gt_images": len(names) - 1,
                         "gt_components_ge_100px": gt_total,
                         "missed_gt_components_lt_10pct": missed_total,
                         "images_with_missed_gt": missed_images,
                         "images_with_exact_top40_missed_hit": hit_missed_images,
                         "missed_touching_48px_margin": edge_touched_missed,
                         "missed_without_gt_pixel_in_allowed_center_area": edge_inaccessible_missed},
        "top_original_rank": totals,
        "true_negative_nuc_005": negative,
        "per_image": per_image,
        "missed_object_details": object_details,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Nuc 暗芯候选在归档内部验证集上的回顾性核查（2026-10-17）",
        "",
        "复用 [现有候选函数](../code/nuc_cross_slice_review.py) 的固定 10/32 px 高斯尺度、60 px 峰间距、48 px 边界排除、每图最多 40 点。对 34 张 `nuc_###` **各自只计算中心图响应**；编号不当作相邻 Z 层。划分与阈值文件的 SHA256 符合归档运行清单，输入和人工 mask 的 SHA256 全部符合冻结划分，34 张归档 selected mask 逐像素等于概率图以 0.45 阈值生成的结果。",
        "",
        f"人工 GT 用 8 连通且面积≥100 px 的连通域，共 {gt_total} 个；其中冻结预测覆盖该域像素不足 10% 的有 {missed_total} 个。{edge_touched_missed} 个漏检域碰到 48 px 排除带；其中 {edge_inaccessible_missed} 个完全没有 GT 像素落在允许的候选中心区域。以下分母始终含全部 {missed_total} 个漏检域。",
        "",
        "| 原始排名 | 全部黄色点 | 掩膜外待复核点 | 精确落入漏检 GT 域 | 10 px 内触及漏检 GT 域 | 掩膜外精确命中 | 掩膜外 10 px 内命中 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cutoff in RANK_CUTOFFS:
        t = totals[str(cutoff)]
        lines.append(f"| ≤{cutoff} | {t['markers']} | {t['outside_prediction_queue']} | "
                     f"{t['exact_missed_hits']}/{missed_total} | {t['within_10px_missed_hits']}/{missed_total} | "
                     f"{t['queue_exact_missed_hits']}/{missed_total} | {t['queue_within_10px_missed_hits']}/{missed_total} |")
    t40 = totals["40"]
    lines += [
        "",
        "命中数是每个阈值下候选点与 GT 连通域的一对一最大匹配。主要定义是**点中心在 GT 域内**；10 px 距离只是预设的定位敏感性检查。掩膜外队列从原排名过滤，不补足 40 点。GT 区域很小或边缘不可达时仍留在分母。",
        "",
        f"{missed_total} 个漏检域分布在 {missed_images} 张图，top40 精确命中分布在 {hit_missed_images} 张图；"
        f"另外 {missed_total - t40['exact_missed_hits']} 个漏检域没有候选点落入，即使放宽到 10 px 也没有新增命中。",
        "",
        f"确认全黑的验证图 `nuc_005.tif` 有 {negative['markers_top40']} 个固定候选点，GT 真核为 0，"
        f"其中 {negative['outside_prediction_queue_top40']} 个点在预测 mask 外；归档模型自身在该图有 {negative['baseline_predicted_positive_pixels']} 个预测阳性像素。"
        "这些候选点相对该图人工 GT 均是假提示，但固定给每图 40 点**不是**分割模型的 40 个假阳性核，二者计数单位也不同。",
        "",
        f"全 34 图 top40 有 {t40['markers']} 点，其中 {t40['outside_prediction_queue']} 点进入掩膜外复核队列；"
        f"{t40['centers_inside_gt']} 点中心落在任意 GT 像素内，掩膜外队列中有 {t40['queue_centers_inside_gt']} 点，"
        f"另有 {t40['queue_centers_outside_gt']} 点中心落在 GT 外。"
        "掩膜外不等于漏检：其余点可能是背景纹理、已有 GT 的边缘，或标注语义不确定处。因漏掉 8 个已知漏检域且阴性图仍产生 39 个掩膜外提示，候选只适合辅助人工逐图查看，不应自动生成或修改标签。",
        "",
        "这只是已有同批次验证集的回顾性诊断；训练和验证有近重复图，当前目标 `fluo_###` 无完整人工 GT。此处数值不能推断目标数据的召回率、精准率或新来源泛化，也不用于调候选参数。中心图核查不评价跨层支持的贡献。",
        "",
        "[可复现脚本](../code/nuc_review_internal_audit.py) · [聚合及逐图 JSON](nuc_dark_core_validation_audit_20261017.json)",
        "",
    ]
    output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
