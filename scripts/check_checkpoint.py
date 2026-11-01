"""Run the repository's real unittest suites and retain machine-readable evidence.

Pytest modules are inventoried, not counted as passing or skipped by unittest.
Run the original pytest suite separately when its dependencies are available.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))
    suite = unittest.TestSuite()
    included, other = [], []
    for path in sorted((root / "tests").glob("test_*.py")):
        parsed = ast.parse(path.read_text())
        is_unittest = any(isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Attribute) and base.attr in ("TestCase", "IsolatedAsyncioTestCase")
            for base in node.bases) for node in ast.walk(parsed))
        if is_unittest:
            included.append(path.name)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(path.stem))
        else:
            other.append(path.name)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"runner": "stdlib-unittest", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "successful": result.wasSuccessful(),
              "included_modules": included, "pytest_modules_not_run_by_this_runner": other}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
