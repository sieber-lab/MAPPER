"""Compound-held-out predictions for a recorded feature configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapper.screen_cli import main

if __name__ == "__main__":
    main("features", leave_one_out=True)
