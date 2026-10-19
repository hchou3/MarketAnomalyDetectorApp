"""Guard against bloated fixture files being committed."""

from __future__ import annotations

import pathlib

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"
_MAX_BYTES = 50 * 1024  # 50 KB


def test_fixture_files_do_not_exceed_50kb():
    oversize = [p for p in _FIXTURE_DIR.rglob("*") if p.is_file() and p.stat().st_size > _MAX_BYTES]
    assert not oversize, (
        "Fixture files exceed 50 KB (copy only the rows the tests need):\n"
        + "\n".join(f"  {p}  ({p.stat().st_size // 1024} KB)" for p in sorted(oversize))
    )
