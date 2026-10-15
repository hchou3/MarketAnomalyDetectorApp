"""mad-data CLI — ingestion and data-management commands."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

_DB_NOT_INIT_MSG = "Database not initialized — run: uv run alembic upgrade head"


def _require_db_ready(engine) -> None:
    """Exit 2 if the instruments table does not exist.

    Called before every database command so the error is clear when the DB
    is reachable but migrations have not been run yet.  If the DB itself is
    unreachable the connection attempt raises an ordinary exception which the
    caller's ``except Exception`` handler catches.
    """
    from sqlalchemy import inspect as sa_inspect

    insp = sa_inspect(engine)
    if not insp.has_table("instruments"):
        print(_DB_NOT_INIT_MSG, file=sys.stderr)
        raise SystemExit(2)


# ── sync-registry ─────────────────────────────────────────────────────────────


def _cmd_sync_registry(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .db.repository import sync_registry
    from .registry import load_registry

    try:
        instruments = load_registry()
    except Exception as exc:
        print(f"ERROR: failed to load registry: {exc}", file=sys.stderr)
        return 1

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            sync_registry(session, instruments)
            session.commit()
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    n_sources = sum(len(i.sources) for i in instruments)
    print(f"sync-registry: {len(instruments)} instruments, {n_sources} sources synced.")
    return 0


# ── load-ews ──────────────────────────────────────────────────────────────────


def _cmd_load_ews(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .providers.ews import load_ews

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            result = load_ews(session, csv_path=args.csv)
            session.commit()
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    print(
        f"load-ews: {result['prices_written']} price rows, "
        f"{result['labels_written']} label rows, "
        f"{result['instruments_ok']} ok, "
        f"{result['instruments_failed']} failed."
    )
    return 1 if result["instruments_failed"] > 0 else 0


# ── backfill ──────────────────────────────────────────────────────────────────


def _cmd_backfill(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .db.repository import add_run_item, finish_run, start_run, sync_registry
    from .ingestion import PostgresPriceSink, run_backfill
    from .providers.cboe import CboeProvider
    from .providers.fred import FredProvider
    from .providers.yahoo import YahooProvider
    from .registry import load_registry

    try:
        instruments = load_registry()
    except Exception as exc:
        print(f"ERROR: failed to load registry: {exc}", file=sys.stderr)
        return 1

    # Filter to requested instruments when --instrument is given
    if args.instrument:
        requested = set(args.instrument)
        instruments = [i for i in instruments if i.id in requested]
        missing = requested - {i.id for i in instruments}
        for m in sorted(missing):
            print(f"ERROR: unknown instrument ID: {m!r}", file=sys.stderr)
        if missing:
            return 1

    # Active instruments only (valid_to not set or in the future)
    today = date.today()
    instruments = [
        i for i in instruments if not i.valid_to or date.fromisoformat(i.valid_to) >= today
    ]

    since: date | None = None
    if args.since:
        try:
            since = date.fromisoformat(args.since)
        except ValueError:
            print(
                f"ERROR: --since must be ISO date (YYYY-MM-DD), got {args.since!r}",
                file=sys.stderr,
            )
            return 1

    # Build Yahoo per-symbol auto_adjust map from the full instrument list
    yahoo_auto_adjust = {
        src.symbol: src.auto_adjust
        for inst in instruments
        for src in inst.sources
        if src.provider == "yahoo" and src.symbol
    }
    providers = {
        "yahoo": YahooProvider(symbol_auto_adjust=yahoo_auto_adjust),
        "fred": FredProvider(),
        "cboe": CboeProvider(),
    }

    try:
        engine = create_engine(url=args.database_url)
    except Exception as exc:
        print(f"ERROR: cannot connect to database: {exc}", file=sys.stderr)
        return 1

    try:
        with Session(engine) as session:
            _require_db_ready(engine)
            sync_registry(session, instruments)
            session.flush()

            run = start_run(session, mode="backfill")
            session.flush()

            sink = PostgresPriceSink(session)
            results = run_backfill(
                instruments,
                providers=providers,
                sink=sink,
                session=session,
                run_id=run.id,
                since=since,
            )

            ok = failed = skipped = total_rows = 0
            for r in results:
                if r.status == "ok":
                    ok += 1
                    total_rows += r.rows_written
                    add_run_item(
                        session,
                        run,
                        r.instrument_id,
                        r.provider,
                        "ok",
                        rows_written=r.rows_written,
                        first_date=r.first_date,
                        last_date=r.last_date,
                    )
                elif r.status == "error":
                    failed += 1
                    add_run_item(
                        session,
                        run,
                        r.instrument_id,
                        r.provider or "unknown",
                        "error",
                        error=r.error,
                    )
                    log.warning("backfill failed: %s — %s", r.instrument_id, r.error)
                else:  # skipped
                    skipped += 1

            run_status = "success" if failed == 0 else ("failed" if ok == 0 else "partial")
            finish_run(
                session,
                run,
                status=run_status,
                instruments_ok=ok,
                instruments_failed=failed,
                rows_written=total_rows,
            )
            session.commit()

    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    print(f"backfill: {ok} ok, {failed} failed, {skipped} skipped, {total_rows} rows written.")
    # Non-zero only when every instrument failed or the DB was unreachable
    return 1 if (ok == 0 and failed > 0) else 0


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="mad-data",
        description="Market Anomaly Detector — data management commands.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── sync-registry ────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "sync-registry",
        help="Sync instruments and symbol_map from config/instruments.yaml.",
    )
    p.add_argument(
        "--database-url",
        default=None,
        metavar="URL",
        help="Database URL (default: MAD_DATABASE_URL env var).",
    )

    # ── load-ews ─────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "load-ews",
        help="Load the Bloomberg EWS CSV into the database (provider=bloomberg_ews).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the EWS CSV (default: data/raw/ews_weekly_2000_2021.csv).",
    )
    p.add_argument(
        "--database-url",
        default=None,
        metavar="URL",
        help="Database URL (default: MAD_DATABASE_URL env var).",
    )

    # ── backfill ─────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "backfill",
        help="Fetch full price history for active instruments and store in the DB.",
    )
    p.add_argument(
        "--instrument",
        action="append",
        metavar="ID",
        help="Canonical instrument ID to backfill (repeatable; default: all active).",
    )
    p.add_argument(
        "--since",
        metavar="DATE",
        help="Fetch data from this ISO date (YYYY-MM-DD) onward (default: full history).",
    )
    p.add_argument(
        "--database-url",
        default=None,
        metavar="URL",
        help="Database URL (default: MAD_DATABASE_URL env var).",
    )

    args = parser.parse_args()
    if args.command == "sync-registry":
        sys.exit(_cmd_sync_registry(args))
    elif args.command == "load-ews":
        sys.exit(_cmd_load_ews(args))
    elif args.command == "backfill":
        sys.exit(_cmd_backfill(args))
    else:
        parser.print_help()
        sys.exit(1)
