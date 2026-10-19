"""Safety checks for same-name Nuc review inference and its output manifest."""

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
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nuc_predict_review import predict_folder, select_inputs


class NucPredictReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_dir = self.root / "fluo"
        self.input_dir.mkdir()
        for index in (1, 2):
            tifffile.imwrite(
                self.input_dir / f"fluo_{index:03d}.tif",
                np.arange(80 * 96, dtype=np.uint16).reshape(80, 96),
            )
        (self.input_dir / "._fluo_001.tif").write_bytes(b"AppleDouble metadata")
        self.model_path = self.root / "nuc_unet.keras"
        self.model_path.write_bytes(b"synthetic checkpoint for mocked inference")

    def test_input_selection_ignores_appledouble_and_limit_is_deterministic(self) -> None:
        self.assertEqual(
            [path.name for path in select_inputs(self.input_dir, 1)],
            ["fluo_001.tif"],
        )
        self.assertEqual(len(select_inputs(self.input_dir, None)), 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            select_inputs(self.input_dir, 0)

    def test_requires_reviewed_checkpoint_and_fresh_separate_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            predict_folder(self.input_dir, self.root / "uncreated", self.model_path)
        self.assertFalse((self.root / "uncreated").exists())
        with patch(
            "nuc_predict_review.MODEL_SHA256",
            hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
        ):
            with self.assertRaisesRegex(ValueError, "separate"):
                predict_folder(self.input_dir, self.input_dir, self.model_path)
            already_there = self.root / "existing"
            already_there.mkdir()
            with self.assertRaises(FileExistsError):
                predict_folder(self.input_dir, already_there, self.model_path)

    def test_complete_folder_writes_binary_masks_and_hash_manifest(self) -> None:
        probability = np.zeros((80, 96), dtype=np.float32)
        probability[12:52, 15:60] = 0.10
        probability[25:45, 30:55] = 0.80
        fake_tf = types.SimpleNamespace(
            keras=types.SimpleNamespace(
                models=types.SimpleNamespace(load_model=lambda *args, **kwargs: object())
            )
        )
        output_dir = self.root / "new_review_masks"
        with (
            patch("nuc_predict_review.MODEL_SHA256", hashlib.sha256(self.model_path.read_bytes()).hexdigest()),
            patch.dict(sys.modules, {"tensorflow": fake_tf}),
            patch("nuc_predict_review.predict_sliding_window", return_value=probability),
        ):
            manifest_path = predict_folder(
                self.input_dir, output_dir, self.model_path, limit=1
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["low_threshold"], 0.03)
        self.assertEqual(manifest["core_threshold"], 0.45)
        self.assertEqual(len(manifest["files"]), 1)
        row = manifest["files"][0]
        self.assertEqual(row["filename"], "fluo_001.tif")
        output = output_dir / "fluo_001.tif"
        saved = tifffile.imread(output)
        self.assertEqual(saved.shape, (80, 96))
        self.assertEqual(saved.dtype, np.uint8)
        self.assertEqual(set(np.unique(saved).tolist()), {0, 255})
        self.assertEqual(row["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(row["height"], 80)
        self.assertEqual(row["width"], 96)
        self.assertFalse((output_dir / "fluo_002.tif").exists())


if __name__ == "__main__":
    unittest.main()
