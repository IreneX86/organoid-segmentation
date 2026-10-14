"""Synthetic, TensorFlow-free checks for the exploratory LD fold ensemble."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_focal_ensemble_predict import load_fold_specs, predict_folder
from tiff_io import normalize_image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class IdentityModel:
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return batch


class LDFocalEnsemblePredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs" / "ld"
        self.runs.mkdir(parents=True)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        self.outputs = self.root / "output"
        self.image = np.arange(15 * 19, dtype=np.uint16).reshape(15, 19)
        tifffile.imwrite(self.inputs / "ld_001.tif", self.image)
        tifffile.imwrite(self.inputs / "ld_002.tif", self.image[::-1])
        (self.inputs / "._ld_003.tif").write_bytes(b"AppleDouble metadata")
        self.thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
        for fold, threshold in enumerate(self.thresholds):
            run = self.runs / f"fold_{fold}"
            run.mkdir()
            (run / "focal_unet.keras").write_bytes(f"model {fold}".encode())
            save_json(run / "threshold.json", {"threshold": threshold})
            save_json(run / "experiment_config.json", {
                "target": "ld", "fold": fold, "num_folds": 5,
                "candidate": "ld_focal_dice_gamma2_v1",
                "loss": "focal_bce_gamma2_plus_soft_dice",
                "patch_size": 512, "overlap": 128,
                "prediction_batch_size": 4,
                "lower_percentile": 1.0, "upper_percentile": 99.8,
            })
            save_json(run / "split.json", {
                "target": "ld", "fold": fold, "num_folds": 5,
            })
            save_json(run / "selection.lock.json", {
                "target": "ld", "fold": fold, "threshold": threshold,
                "checkpoint_sha256": sha256(run / "focal_unet.keras"),
                "threshold_sha256": sha256(run / "threshold.json"),
                "experiment_config_sha256": sha256(run / "experiment_config.json"),
                "split_sha256": sha256(run / "split.json"),
            })

    def test_five_votes_preserve_original_grid_and_skip_complete_run(self) -> None:
        loaded: list[str] = []
        cleared: list[int] = []

        def loader(checkpoint: Path) -> IdentityModel:
            loaded.append(checkpoint.parent.name)
            return IdentityModel()

        manifest = predict_folder(
            self.runs, self.inputs, self.outputs,
            model_loader=loader, cleanup_model=lambda: cleared.append(1),
        )
        self.assertEqual(loaded, [f"fold_{fold}" for fold in range(5)])
        self.assertEqual(len(cleared), 5)
        self.assertEqual(manifest["image_count"], 2)
        self.assertEqual(manifest["ignored_appledouble_files"], 1)
        self.assertTrue(manifest["not_validated_as_ensemble"])
        self.assertEqual(len(manifest["folds"]), 5)
        self.assertFalse((self.outputs / "_votes").exists())
        self.assertFalse((self.outputs / "._ld_003.tif").exists())
        for name, raw in (("ld_001.tif", self.image), ("ld_002.tif", self.image[::-1])):
            mask = tifffile.imread(self.outputs / name)
            expected = (normalize_image(raw, 1.0, 99.8) >= 0.4).astype(np.uint8) * 255
            self.assertEqual(mask.shape, raw.shape)
            self.assertEqual(mask.dtype, np.uint8)
            np.testing.assert_array_equal(mask, expected)
        repeated = predict_folder(
            self.runs, self.inputs, self.outputs,
            model_loader=lambda _: self.fail("A completed run must not load a model"),
        )
        self.assertEqual(repeated, manifest)

    def test_partial_directory_never_overwritten(self) -> None:
        self.outputs.mkdir()
        existing = self.outputs / "ld_001.tif"
        existing.write_bytes(b"preexisting result")
        with self.assertRaises(FileNotFoundError):
            predict_folder(self.runs, self.inputs, self.outputs, model_loader=IdentityModel)
        self.assertEqual(existing.read_bytes(), b"preexisting result")

    def test_changed_input_output_or_checkpoint_fails_closed(self) -> None:
        predict_folder(self.runs, self.inputs, self.outputs, model_loader=lambda _: IdentityModel())
        mask_path = self.outputs / "ld_001.tif"
        mask_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "failed verification"):
            predict_folder(self.runs, self.inputs, self.outputs, model_loader=IdentityModel)

        # Restore the complete result using its original bytes so the next
        # checks isolate source-input and checkpoint provenance.
        self.outputs.rename(self.root / "tampered_output")
        predict_folder(self.runs, self.inputs, self.outputs, model_loader=lambda _: IdentityModel())
        tifffile.imwrite(self.inputs / "ld_001.tif", self.image + 1)
        with self.assertRaisesRegex(ValueError, "failed verification"):
            predict_folder(self.runs, self.inputs, self.outputs, model_loader=IdentityModel)

        tifffile.imwrite(self.inputs / "ld_001.tif", self.image)
        (self.runs / "fold_2" / "focal_unet.keras").write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError, "selection lock"):
            predict_folder(self.runs, self.inputs, self.outputs, model_loader=IdentityModel)

    def test_threshold_lock_and_scientific_settings_are_required(self) -> None:
        run = self.runs / "fold_4"
        (run / "threshold.json").write_text('{"threshold":0.9}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selection lock"):
            load_fold_specs(self.runs)


if __name__ == "__main__":
    unittest.main()
