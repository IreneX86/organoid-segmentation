"""Guard the Fluo-only training bundle against ER-channel and split leakage."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er_fluo_student import prepare, sha256, TEST_CELL, VALIDATION_CELL, weighted_bce_dice_5_loss  # noqa: E402


def test_prepare_contains_only_fluo_and_masks(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    masks = tmp_path / "masks"
    names = [VALIDATION_CELL, TEST_CELL] + [f"cell_train_{i}" for i in range(6)]
    for index, name in enumerate(names):
        relative = Path("domain_a" if index % 2 else "domain_b") / name
        source = inputs / relative
        target = masks / relative
        source.mkdir(parents=True)
        target.mkdir(parents=True)
        image = np.full((2, 32, 32), index + 5, dtype=np.uint16)
        label = np.zeros((2, 32, 32), dtype=np.uint8)
        label[:, 5:10, 5:10] = 255
        tifffile.imwrite(source / "ER.tif", image, imagej=True)
        tifffile.imwrite(source / "fluo.tif", image, imagej=True)
        tifffile.imwrite(target / "ER_mask_line.tif", label, imagej=True)
        (target / "manifest.json").write_text(json.dumps({
            "source_z_start_zero_based": 0, "source_z_stop_exclusive": 2,
            "source_shape": [2, 32, 32],
            "source_fluo_sha256": sha256(source / "fluo.tif"),
            "outputs": {"ER_mask_line": {"sha256": sha256(target / "ER_mask_line.tif")}},
        }))

    archive = tmp_path / "student.tar.gz"
    prepare(inputs, masks, archive)
    with tarfile.open(archive) as reader:
        entries = set(reader.getnames())
        assert all("ER.tif" not in entry for entry in entries)
        assert len([entry for entry in entries if entry.endswith("fluo.tif")]) == 8
        assert len([entry for entry in entries if entry.endswith("mask.tif")]) == 8
        manifest = json.load(reader.extractfile("dataset_manifest.json"))
    rows = manifest["records"]
    assert {row["cell"] for row in rows if row["split"] == "validation"} == {VALIDATION_CELL}
    assert {row["cell"] for row in rows if row["split"] == "test"} == {TEST_CELL}
    assert len([row for row in rows if row["split"] == "train"]) == 6
    assert "_local_fluo" not in json.dumps(manifest)


def test_weighted_loss_emphasizes_missed_positive_and_has_finite_gradient() -> None:
    import pytest
    tf = pytest.importorskip("tensorflow")
    from losses import binary_crossentropy_dice_loss

    truth = tf.constant([[[[1.0], [0.0]], [[0.0], [0.0]]]])
    prediction = tf.Variable([[[[0.1], [0.1]], [[0.1], [0.1]]]])
    with tf.GradientTape() as tape:
        new_loss = weighted_bce_dice_5_loss(truth, prediction)
    gradient = tape.gradient(new_loss, prediction)
    assert float(new_loss) > float(binary_crossentropy_dice_loss(truth, prediction))
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)))
    assert float(gradient[0, 0, 0, 0]) < 0  # raise missed-positive probability
