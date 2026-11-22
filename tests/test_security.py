"""Guard rails against credentials and personal data leaking into the repo."""

import re
from pathlib import Path

import pytest

from cropmon.models.deep import build_hybrid_cnn_lstm

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ["src", "dashboard", "scripts", "tests"]
SKIP_SUFFIXES = {".pyc", ".png", ".joblib", ".npz"}

PHONE = re.compile(r"\+91[0-9]{8,12}")
TWILIO_SID = re.compile(r"\bAC[0-9a-fA-F]{32}\b")
TWILIO_KEY = re.compile(r"\bSK[0-9a-fA-F]{32}\b")
PLACEHOLDER = re.compile(r"YOUR_TWILIO|YOUR_FAST2SMS|your_api_key", re.IGNORECASE)


def _text_files():
    for folder in SCANNED:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix not in SKIP_SUFFIXES and path != Path(__file__):
                yield path


@pytest.mark.parametrize("pattern,name", [(PHONE, "phone number"), (TWILIO_SID, "Twilio SID"), (TWILIO_KEY, "Twilio key")])
def test_no_hardcoded_credentials(pattern, name):
    offenders = [str(path.relative_to(ROOT)) for path in _text_files() if pattern.search(path.read_text(errors="ignore"))]
    assert not offenders, f"{name} committed in {offenders}"


def test_no_placeholder_credentials():
    offenders = [
        str(path.relative_to(ROOT)) for path in _text_files() if PLACEHOLDER.search(path.read_text(errors="ignore"))
    ]
    assert not offenders, f"placeholder credentials in {offenders}"


def test_deep_module_explains_missing_dependency():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="deep"):
            build_hybrid_cnn_lstm()
    else:
        pytest.skip("TensorFlow installed; optional builder is importable")
