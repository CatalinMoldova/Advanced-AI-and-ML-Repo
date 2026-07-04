"""Dependency contract tests for runtime imports used by the training code."""

from __future__ import annotations

import pathlib
import re
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def requirement_lower_bound(package_name: str) -> tuple[int, ...]:
    requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()
    pattern = re.compile(rf"^{re.escape(package_name)}\s*>=\s*([0-9]+(?:\.[0-9]+)*)\s*$")

    for requirement in requirements:
        match = pattern.match(requirement.strip())
        if match:
            return tuple(int(part) for part in match.group(1).split("."))

    raise AssertionError(f"{package_name!r} must declare an explicit >= lower bound")


class DependencyContractTests(unittest.TestCase):
    def test_trl_lower_bound_includes_sft_config(self) -> None:
        self.assertGreaterEqual(requirement_lower_bound("trl"), (0, 9, 2))


if __name__ == "__main__":
    unittest.main()
