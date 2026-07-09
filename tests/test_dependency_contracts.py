"""Dependency contract checks for imported third-party APIs."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DependencyContractTests(unittest.TestCase):
    def test_trl_lower_bound_supports_sft_config(self):
        requirements = (REPO_ROOT / "requirements.txt").read_text()
        match = re.search(r"^trl>=([0-9]+)\.([0-9]+)\.([0-9]+)$", requirements, re.MULTILINE)

        self.assertIsNotNone(match, "requirements.txt must pin a TRL lower bound")
        lower_bound = tuple(int(part) for part in match.groups())
        self.assertGreaterEqual(lower_bound, (0, 9, 2))


if __name__ == "__main__":
    unittest.main()
