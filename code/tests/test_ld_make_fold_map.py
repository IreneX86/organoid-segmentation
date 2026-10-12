"""Keep input-only suspect image groups intact in the frozen LD split."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ld_make_fold_map import _groups_from_edges


class LDSimilarityGroupTests(unittest.TestCase):
    def test_connected_edges_form_one_group(self) -> None:
        names = ["a.tif", "b.tif", "c.tif", "d.tif", "e.tif"]
        edges = [
            {"left": "a.tif", "right": "b.tif"},
            {"left": "b.tif", "right": "c.tif"},
            {"left": "d.tif", "right": "e.tif"},
        ]
        self.assertEqual(
            _groups_from_edges(names, edges),
            {frozenset({"a.tif", "b.tif", "c.tif"}), frozenset({"d.tif", "e.tif"})},
        )


if __name__ == "__main__":
    unittest.main()
