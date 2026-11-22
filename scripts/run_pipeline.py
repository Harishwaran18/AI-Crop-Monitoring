#!/usr/bin/env python
"""Run the full analysis pipeline and write artifacts plus figures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropmon.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pipeline", *sys.argv[1:]]))
