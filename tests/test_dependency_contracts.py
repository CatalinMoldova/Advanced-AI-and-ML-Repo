"""Dependency constraints required by the training code."""

from __future__ import annotations

import ast
import re
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
TRAINING_SOURCE = REPO_ROOT / "src" / "train_lora.py"


def _requirement_lower_bound(package_name: str) -> tuple[int, ...] | None:
    pattern = re.compile(
        rf"^\s*{re.escape(package_name)}\s*>=\s*([0-9]+(?:\.[0-9]+)*)\s*(?:#.*)?$",
        flags=re.IGNORECASE,
    )

    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return tuple(int(part) for part in match.group(1).split("."))
    return None


def _imports_symbol(module_name: str, symbol_name: str) -> bool:
    tree = ast.parse(TRAINING_SOURCE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != module_name:
            continue
        if any(alias.name == symbol_name for alias in node.names):
            return True
    return False


class DependencyContractTests(unittest.TestCase):
    def test_trl_lower_bound_supports_sft_config_import(self) -> None:
        if not _imports_symbol("trl", "SFTConfig"):
            self.skipTest("src.train_lora no longer imports trl.SFTConfig")

        lower_bound = _requirement_lower_bound("trl")
        self.assertIsNotNone(lower_bound, "requirements.txt must constrain trl")
        self.assertGreaterEqual(
            lower_bound,
            (0, 9, 2),
            "trl.SFTConfig is not available in the previously allowed 0.8.x releases",
        )


if __name__ == "__main__":
    unittest.main()
