"""Dependency contract checks for APIs imported by the source tree."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DependencyContractsTest(unittest.TestCase):
    def test_trl_floor_supports_sft_config_import(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text().splitlines()
        trl_requirement = next(
            line.strip()
            for line in requirements
            if line.strip().startswith("trl>=")
        )

        floor = trl_requirement.split(">=", 1)[1]
        self.assertGreaterEqual(
            tuple(map(int, floor.split("."))),
            (0, 9, 2),
            "src.train_lora imports trl.SFTConfig, which is unavailable in TRL 0.8.x",
        )

    def test_train_lora_imports_sft_config_from_trl(self) -> None:
        tree = ast.parse((ROOT / "src" / "train_lora.py").read_text())
        trl_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "trl"
            for alias in node.names
        }

        self.assertIn("SFTConfig", trl_imports)


if __name__ == "__main__":
    unittest.main()
