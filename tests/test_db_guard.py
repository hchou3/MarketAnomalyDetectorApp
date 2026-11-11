"""Unit tests for the DB fixture safety guards in conftest.py.

These tests are non-DB (no MAD_TEST_DATABASE_URL required) because they exercise
the pure URL-parsing and live-connection check logic, not the fixture itself.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest
from conftest import _check_live_db, _check_test_url, _db_name

# ── _db_name ──────────────────────────────────────────────────────────────────


def test_db_name_extracts_database():
    assert _db_name("postgresql+psycopg://u:p@localhost:5432/mydb") == "mydb"


def test_db_name_handles_no_database():
    assert _db_name("postgresql+psycopg://u:p@localhost:5432/") == ""


# ── _check_test_url ───────────────────────────────────────────────────────────


def test_guard_accepts_valid_test_url():
    """A test URL with a _test suffix and different db name must pass."""
    _check_test_url(
        "postgresql+psycopg://mad:mad@localhost:5432/mad_test",
        "postgresql+psycopg://mad:mad@localhost:5432/mad",
    )


def test_guard_rejects_same_db_name_via_different_host():
    """127.0.0.1 vs localhost used to bypass the old string-equality check."""
    with pytest.raises(pytest.fail.Exception, match="database name"):
        _check_test_url(
            "postgresql+psycopg://mad:mad@127.0.0.1:5432/mad",
            "postgresql+psycopg://mad:mad@localhost:5432/mad",
        )


def test_guard_rejects_same_db_name_via_different_driver():
    """postgresql:// vs postgresql+psycopg:// used to bypass the old check."""
    with pytest.raises(pytest.fail.Exception, match="database name"):
        _check_test_url(
            "postgresql://mad:mad@localhost:5432/mad",
            "postgresql+psycopg://mad:mad@localhost:5432/mad",
        )


def test_guard_rejects_same_db_name_with_trailing_option():
    """A query-string option on the test URL used to bypass the old check."""
    with pytest.raises(pytest.fail.Exception, match="database name"):
        _check_test_url(
            "postgresql+psycopg://mad:mad@localhost:5432/mad?sslmode=disable",
            "postgresql+psycopg://mad:mad@localhost:5432/mad",
        )


def test_guard_rejects_name_without_test_suffix():
    """Database name 'staging' (no _test) must be refused even if names differ."""
    with pytest.raises(pytest.fail.Exception, match="_test"):
        _check_test_url(
            "postgresql+psycopg://mad:mad@localhost:5432/staging",
            "postgresql+psycopg://mad:mad@localhost:5432/mad",
        )


def test_guard_rejects_name_without_test_suffix_different_host():
    """No _test suffix should be caught regardless of host differences."""
    with pytest.raises(pytest.fail.Exception, match="_test"):
        _check_test_url(
            "postgresql+psycopg://mad:mad@127.0.0.1:5432/mad_ci",
            "postgresql+psycopg://mad:mad@localhost:5432/mad",
        )


def test_guard_check_is_case_insensitive():
    """Database name comparison is case-insensitive."""
    with pytest.raises(pytest.fail.Exception, match="database name"):
        _check_test_url(
            "postgresql+psycopg://mad:mad@localhost:5432/MAD",
            "postgresql+psycopg://mad:mad@localhost:5432/mad",
        )


# ── _check_live_db ────────────────────────────────────────────────────────────


def test_guard_live_db_rejects_non_test_name():
    """_check_live_db raises RuntimeError when current_database() lacks _test."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = lambda s: conn
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.scalar.return_value = "mad"

    with pytest.raises(RuntimeError, match="_test"):
        _check_live_db(engine)


def test_guard_live_db_accepts_test_name():
    """_check_live_db passes silently when current_database() ends with _test."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = lambda s: conn
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.scalar.return_value = "mad_test"

    _check_live_db(engine)  # must not raise


def test_guard_live_db_rejects_none_database():
    """_check_live_db rejects a None result from current_database()."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = lambda s: conn
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.scalar.return_value = None

    with pytest.raises(RuntimeError, match="_test"):
        _check_live_db(engine)


# ── Autouse _block_app_db sentinel checks ────────────────────────────────────


def test_mad_database_url_is_sentinel_during_tests():
    """The _block_app_db autouse fixture must override MAD_DATABASE_URL in every test."""
    from market_anomaly_detector.db.engine import _DEFAULT_DATABASE_URL, get_database_url

    url = get_database_url()
    assert url != _DEFAULT_DATABASE_URL, (
        f"get_database_url() returned the default app URL ({_DEFAULT_DATABASE_URL!r}). "
        "The _block_app_db autouse fixture must be active for every test."
    )
    assert "127.0.0.1:1" in url, f"Expected sentinel URL, got {url!r}"


def test_no_handler_opens_app_db_url(monkeypatch):
    """A handler called with database_url=None must use the sentinel, not the app DB."""
    import market_anomaly_detector.db.engine as _eng

    opened: list[str] = []

    def _spy(url, **kw):
        opened.append(str(url))
        raise OSError("connection blocked by test spy")

    monkeypatch.setattr(_eng.sa, "create_engine", _spy)

    import market_anomaly_detector.cli as _cli

    # _cmd_sync_registry with database_url=None resolves via get_database_url(),
    # which must return the sentinel set by the autouse fixture.
    _cli._cmd_sync_registry(argparse.Namespace(database_url=None, allow_symbol_change=False))

    assert opened, "expected spy to record at least one create_engine call"
    assert not any(u == _eng._DEFAULT_DATABASE_URL for u in opened), (
        f"Handler opened the default app database URL: {opened}"
    )
