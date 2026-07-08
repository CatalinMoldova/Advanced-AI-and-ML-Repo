"""Dependency constraints required by the training code."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
TRAINING_MODULE = REPO_ROOT / "src" / "train_lora.py"


def _parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class DependencyContractsTest(unittest.TestCase):
    def test_trl_lower_bound_supports_sft_config(self) -> None:
        source = TRAINING_MODULE.read_text()
        if "SFTConfig" not in source:
            self.skipTest("train_lora.py no longer uses TRL's SFTConfig")

        requirements = REQUIREMENTS.read_text().splitlines()
        trl_line = next(
            (line.strip() for line in requirements if line.strip().startswith("trl")),
            "",
        )

        self.assertTrue(trl_line, "requirements.txt must declare a TRL dependency")
        match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", trl_line)
        self.assertIsNotNone(match, "TRL must have an explicit lower bound")
        self.assertGreaterEqual(
            _parse_version(match.group(1)),
            (0, 9, 2),
            "SFTConfig is not exported by TRL before 0.9.2",
        )


if __name__ == "__main__":
    unittest.main()
