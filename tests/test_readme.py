"""The README must not invent a result, and it must carry the clinical disclaimer."""

from __future__ import annotations

import re
from pathlib import Path


def test_results_section_is_tbd_and_disclaimer_is_present():
    readme = Path("README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "not a medical device" in lowered
    assert "download" in lowered
    assert "gpu" in lowered
    assert "deploy" in lowered
    results = readme.split("## Results", 1)[1].split("\n## ", 1)[0]
    assert "TBD" in results
    assert re.search(r"\d\.\d", results) is None


def test_license_is_mit():
    text = Path("LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 SomkeneA" in text
