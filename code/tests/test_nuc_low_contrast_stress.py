"""Small deterministic checks for the locked Nuc synthetic stress report."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nuc_low_contrast_stress as stress
from tiff_io import ImageLabelPair


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NucLowContrastStressTests(unittest.TestCase):
    def test_stress_uses_fixed_full_image_blur_and_alpha(self) -> None:
        image = np.array([[0.0, 0.4], [0.8, 1.0]], dtype=np.float32)
        blur = np.full(image.shape, 0.25, dtype=np.float32)
        helper = types.SimpleNamespace(gaussian_blur_reflect=lambda x, sigma: blur)
        with patch.dict(sys.modules, {"patch_dataset": helper}):
            result = stress._stress_image(image)
        np.testing.assert_array_equal(result, blur + 0.5 * (image - blur))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, image.shape)

    def test_predict_uses_fixed_cutoff_and_keeps_empty_mask_separate(self) -> None:
        pairs = [
            ImageLabelPair(Path("nuc_005.tif"), Path("nuc_005.tif")),
            ImageLabelPair(Path("nuc_006.tif"), Path("nuc_006.tif")),
        ]
        image = np.ones((2, 2), dtype=np.float32)
        truth = {
            "nuc_005.tif": np.zeros((2, 2), dtype=np.uint8),
            "nuc_006.tif": np.array([[1, 0], [0, 0]], dtype=np.uint8),
        }
        probabilities = iter([
            np.array([[0.50, 0.49], [0.0, 0.0]], dtype=np.float32),
            np.array([[0.50, 0.51], [0.0, 0.0]], dtype=np.float32),
        ])
        recorded: list[tuple[int, int, int]] = []

        def fake_predict(model: object, stressed: np.ndarray, **kwargs: int) -> np.ndarray:
            recorded.append((kwargs["tile_size"], kwargs["overlap"],
                             kwargs["prediction_batch_size"]))
            np.testing.assert_array_equal(stressed, image)
            return next(probabilities)

        with (patch.object(stress, "read_tiff_2d", return_value=image),
              patch.object(stress, "read_binary_mask",
                           side_effect=lambda path: truth[path.name]),
              patch.object(stress, "normalize_image", return_value=image),
              patch.object(stress, "_stress_image", return_value=image),
              patch.object(stress, "predict_sliding_window", side_effect=fake_predict)):
            result = stress._predict_model(object(), pairs)
        self.assertEqual(result["nuc_005.tif"], {
            "truth_group": "true_negative", "false_positive_pixels": 1,
        })
        self.assertEqual(result["nuc_006.tif"]["dice"], 2 / 3)
        self.assertEqual(recorded, [(512, 128, 4), (512, 128, 4)])

    def test_combination_excludes_true_negative_from_overlap_metrics(self) -> None:
        names = [f"nuc_{index:03d}.tif" for index in range(6, 39)]
        baseline = {
            name: {"truth_group": "nonempty", "false_positive_pixels": 2,
                   **{metric: 0.8 for metric in stress.METRICS}}
            for name in names
        }
        candidate = {
            name: {"truth_group": "nonempty", "false_positive_pixels": 1,
                   **{metric: 0.9 for metric in stress.METRICS}}
            for name in names
        }
        baseline["nuc_005.tif"] = {
            "truth_group": "true_negative", "false_positive_pixels": 19,
        }
        candidate["nuc_005.tif"] = {
            "truth_group": "true_negative", "false_positive_pixels": 3,
        }
        rows, positive, negative = stress._combine_results(baseline, candidate)
        self.assertEqual(len(rows), 34)
        self.assertEqual(positive["image_count"], 33)
        self.assertAlmostEqual(positive["dice"]["paired_delta"]["mean"], 0.1)
        self.assertEqual(positive["dice"]["images_improved"], 33)
        self.assertEqual(negative["paired_false_positive_pixels_delta"], -16)
        empty = next(row for row in rows if row["filename"] == "nuc_005.tif")
        self.assertEqual(empty["baseline_dice"], "")
        self.assertEqual(empty["paired_dice_delta"], "")
        del candidate[names[-1]]
        with self.assertRaisesRegex(ValueError, "filenames differ"):
            stress._combine_results(baseline, candidate)

    def test_provenance_rejects_wrong_candidate_science_or_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "frozen_split.json"
            baseline = root / "baseline.keras"
            candidate_dir = root / "candidate"
            candidate_dir.mkdir()
            candidate = candidate_dir / "nuc_unet.keras"
            split.write_bytes(b"frozen-split")
            baseline.write_bytes(b"baseline-checkpoint")
            candidate.write_bytes(b"candidate-checkpoint")
            (candidate_dir / "split.json").write_bytes(split.read_bytes())
            run_manifest = {
                "target": "nuc", "purpose": "same_batch_internal_diagnostic",
                "split_sha256": stress._sha256(split),
                "checkpoint_sha256": stress._sha256(candidate),
            }
            (candidate_dir / "run_manifest.json").write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )
            config = {
                "split_manifest_sha256": stress._sha256(split),
                "target": "nuc", "purpose": "same_batch_internal_diagnostic",
                "architecture": "unet_2d_baseline", "loss": "bce_dice",
                "patch_size": 512, "overlap": 128,
                "batch_size": 4, "steps_per_epoch": 100, "epochs": 100,
                "learning_rate": 1e-4, "positive_patch_probability": 0.6,
                "seed": 42,
                "lower_percentile": 1.0, "upper_percentile": 99.8,
                "prediction_batch_size": 4,
                "checkpoint_selection": "nonempty_validation_full_image_dice_at_0.5",
                "threshold_selection": "nonempty_validation_mean_per_image_dice",
                "low_contrast_probability": 0.5,
                "low_contrast_sigma": 64.0,
                "low_contrast_alpha_min": 0.5,
                "low_contrast_alpha_max": 1.0,
                "low_contrast_boundary": "reflect",
                "spatial_illumination_probability": 0.0,
                "spatial_illumination_max_offset": 0.0,
            }
            config_path = candidate_dir / "experiment_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with (patch.object(stress, "FROZEN_SPLIT_SHA256", stress._sha256(split)),
                  patch.object(stress, "FROZEN_BASELINE_SHA256",
                               stress._sha256(baseline))):
                result = stress._validate_provenance(split, baseline, candidate)
                self.assertEqual(result["candidate_checkpoint_sha256"],
                                 stress._sha256(candidate))
                config["low_contrast_sigma"] = 32.0
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "locked protocol"):
                    stress._validate_provenance(split, baseline, candidate)
                config["low_contrast_sigma"] = 64.0
                config_path.write_text(json.dumps(config), encoding="utf-8")
                candidate.write_bytes(b"changed-checkpoint")
                with self.assertRaisesRegex(ValueError, "provenance"):
                    stress._validate_provenance(split, baseline, candidate)

    def test_output_cannot_be_inside_source_data(self) -> None:
        project = Path("/tmp/synthetic_nuc_project")
        with self.assertRaisesRegex(ValueError, "outside source data"):
            stress.run_stress(
                project, Path("missing.json"), Path("missing-baseline.keras"),
                Path("missing-candidate.keras"),
                project / "label_ref" / "Nuc" / "dangerous",
            )


if __name__ == "__main__":
    unittest.main()
