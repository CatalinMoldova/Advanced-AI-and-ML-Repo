"""Dependency contract tests for APIs imported by the training scripts."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class DependencyContractsTest(unittest.TestCase):
    def test_trl_requirement_supports_sft_config_import(self) -> None:
        train_source = (REPO_ROOT / "src" / "train_lora.py").read_text()
        self.assertIn("from trl import SFTConfig", train_source)

        requirements = (REPO_ROOT / "requirements.txt").read_text()
        match = re.search(r"^trl>=([0-9]+(?:\.[0-9]+)*)$", requirements, re.MULTILINE)
        self.assertIsNotNone(match, "requirements.txt must declare a trl lower bound")

        minimum_version = _version_tuple(match.group(1))
        self.assertGreaterEqual(
            minimum_version,
            (0, 9, 2),
            "SFTConfig is imported by src.train_lora and is not exported by trl 0.8.x",
        )


if __name__ == "__main__":
    unittest.main()
