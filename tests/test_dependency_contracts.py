"""Dependency contract tests for import-time requirements."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_lower_bound(requirement_name: str) -> tuple[int, ...]:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
    pattern = re.compile(rf"^\s*{re.escape(requirement_name)}\s*>=\s*([0-9]+(?:\.[0-9]+)*)\s*$")

    for requirement in requirements:
        match = pattern.match(requirement)
        if match:
            return tuple(int(part) for part in match.group(1).split("."))

    raise AssertionError(f"{requirement_name}>=" " requirement is missing")


class DependencyContractTests(unittest.TestCase):
    def test_trl_lower_bound_covers_train_lora_imports(self) -> None:
        train_lora = ast.parse((PROJECT_ROOT / "src" / "train_lora.py").read_text())
        trl_imports = {
            alias.name
            for node in ast.walk(train_lora)
            if isinstance(node, ast.ImportFrom) and node.module == "trl"
            for alias in node.names
        }

        self.assertIn("SFTConfig", trl_imports)
        self.assertGreaterEqual(parse_lower_bound("trl"), (0, 9, 2))


if __name__ == "__main__":
    unittest.main()
