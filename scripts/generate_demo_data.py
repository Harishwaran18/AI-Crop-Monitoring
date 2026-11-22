#!/usr/bin/env python
"""Write raw simulated spectral cube, sensor CSV, and zone history."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropmon.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo-data", *sys.argv[1:]]))
