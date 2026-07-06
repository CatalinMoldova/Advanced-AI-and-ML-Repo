"""Dependency contracts for public script imports."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


class DependencyContractTests(unittest.TestCase):
    def test_trl_lower_bound_exports_sft_config(self) -> None:
        """src.train_lora imports SFTConfig, which TRL only exports from 0.9.2."""
        requirements = Path("requirements.txt").read_text()
        match = re.search(r"^trl>=([0-9]+)\.([0-9]+)\.([0-9]+)$", requirements, re.MULTILINE)

        self.assertIsNotNone(match, "requirements.txt must pin a TRL lower bound")
        lower_bound = tuple(int(part) for part in match.groups())
        self.assertGreaterEqual(lower_bound, (0, 9, 2))


if __name__ == "__main__":
    unittest.main()
