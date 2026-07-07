"""Dependency contracts for APIs imported by the training scripts."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class DependencyContractTests(unittest.TestCase):
    def test_trl_lower_bound_includes_sft_config(self) -> None:
        train_lora = (ROOT / "src" / "train_lora.py").read_text()
        self.assertIn("from trl import SFTConfig", train_lora)

        requirements = (ROOT / "requirements.txt").read_text()
        match = re.search(r"^trl>=(\d+(?:\.\d+)*)$", requirements, re.MULTILINE)
        self.assertIsNotNone(match, "requirements.txt must pin a TRL lower bound")
        self.assertGreaterEqual(_version_tuple(match.group(1)), (0, 9, 2))


if __name__ == "__main__":
    unittest.main()
