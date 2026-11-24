"""Run the current architecture's regression suite without pytest."""
from pathlib import Path
import unittest


def suite():
    root = Path(__file__).resolve().parent
    return unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(0 if result.wasSuccessful() else 1)
