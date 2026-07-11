"""Dependency constraints required by the training code."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DependencyContractTests(unittest.TestCase):
    def test_trl_constraint_includes_sft_config(self):
        trl_lines = [
            line.strip()
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip().lower().startswith("trl")
        ]

        self.assertEqual(len(trl_lines), 1)
        self.assertRegex(
            trl_lines[0],
            r"^trl\s*>=\s*0\.9\.2$",
            "src.train_lora imports SFTConfig, which TRL added after 0.8.x",
        )

    def test_train_lora_imports_sft_config(self):
        tree = ast.parse((ROOT / "src" / "train_lora.py").read_text())
        imported_from_trl = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "trl"
            for alias in node.names
        }

        self.assertIn("SFTConfig", imported_from_trl)


if __name__ == "__main__":
    unittest.main()
