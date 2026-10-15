"""Run public synthetic package checks without an editable live-source install."""
from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    if sys.platform == "win32":
        git_dir = Path("C:/Program Files/Git/cmd")
        if not (git_dir / "git.exe").is_file():
            raise RuntimeError("The Windows publication-guard test requires Git for Windows")
        os.environ["PATH"] = str(git_dir) + os.pathsep + os.environ.get("PATH", "")
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
