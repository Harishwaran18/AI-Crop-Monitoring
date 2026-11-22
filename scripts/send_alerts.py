#!/usr/bin/env python
"""List alerts, and deliver them when --send is passed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropmon.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["alerts", *sys.argv[1:]]))
