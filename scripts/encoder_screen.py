"""Fit one of the nine recorded augmentation/encoder configurations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapper.screen_cli import main

if __name__ == "__main__":
    main("encoders")
