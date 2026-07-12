"""Dependency contracts for import-time APIs used by the training helpers."""

from __future__ import annotations

import ast
import pathlib
import re
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class DependencyContractTests(unittest.TestCase):
    def test_trl_requirement_supports_sft_config_import(self) -> None:
        train_lora = REPO_ROOT / "src" / "train_lora.py"
        tree = ast.parse(train_lora.read_text())
        imports_sft_config = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "trl"
            and any(alias.name == "SFTConfig" for alias in node.names)
            for node in ast.walk(tree)
        )
        self.assertTrue(imports_sft_config, "src.train_lora should import TRL SFTConfig")

        requirements = (REPO_ROOT / "requirements.txt").read_text()
        match = re.search(r"^trl>=([0-9]+)\.([0-9]+)\.([0-9]+)$", requirements, re.MULTILINE)
        self.assertIsNotNone(match, "requirements.txt must pin a minimum TRL version")

        lower_bound = tuple(int(part) for part in match.groups())
        self.assertGreaterEqual(
            lower_bound,
            (0, 9, 2),
            "TRL versions before 0.9.2 do not export SFTConfig, causing src.train_lora to crash at import time",
        )


if __name__ == "__main__":
    unittest.main()
