"""Fit one of the 192 recorded model/feature configurations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapper.screen_cli import main

if __name__ == "__main__":
    main("features")
