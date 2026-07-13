"""Dependency contract checks for APIs used at import time."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class DependencyContractTests(unittest.TestCase):
    def test_trl_minimum_supports_sft_config(self) -> None:
        requirements = (REPO_ROOT / "requirements.txt").read_text()
        train_source = (REPO_ROOT / "src" / "train_lora.py").read_text()

        self.assertIn("from trl import SFTConfig", train_source)

        match = re.search(r"^trl\s*>=\s*([0-9]+(?:\.[0-9]+){0,2})\s*$", requirements, re.MULTILINE)
        self.assertIsNotNone(match, "requirements.txt must declare a lower bound for trl")

        minimum = _version_tuple(match.group(1))
        self.assertGreaterEqual(
            minimum,
            (0, 9, 2),
            "src.train_lora imports SFTConfig, which is not available in TRL 0.8.x",
        )


if __name__ == "__main__":
    unittest.main()
