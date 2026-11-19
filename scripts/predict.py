"""Predict with the final reference model and its matching score cutoff."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapper.reference import main

if __name__ == "__main__":
    main()
