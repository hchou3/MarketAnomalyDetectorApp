"""mad-data CLI — ingestion and data-management commands."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
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


def _build_providers(instruments):
    """Build the providers dict from active instruments' symbol_auto_adjust config."""
    from .providers.cboe import CboeProvider
    from .providers.fred import FredProvider
    from .providers.yahoo import YahooProvider

    yahoo_auto_adjust = {
        src.symbol: src.auto_adjust
        for inst in instruments
        for src in inst.sources
        if src.provider == "yahoo" and src.symbol
    }
    return {
        "yahoo": YahooProvider(symbol_auto_adjust=yahoo_auto_adjust),
        "fred": FredProvider(),
        "cboe": CboeProvider(),
    }


def _active_instruments(instruments, today=None):
    if today is None:
        today = date.today()
    return [i for i in instruments if not i.valid_to or date.fromisoformat(i.valid_to) >= today]


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
    from .registry import load_registry

    try:
        instruments = load_registry()
    except Exception as exc:
        print(f"ERROR: failed to load registry: {exc}", file=sys.stderr)
        return 1

    if args.instrument:
        requested = set(args.instrument)
        instruments = [i for i in instruments if i.id in requested]
        missing = requested - {i.id for i in instruments}
        for m in sorted(missing):
            print(f"ERROR: unknown instrument ID: {m!r}", file=sys.stderr)
        if missing:
            return 1

    instruments = _active_instruments(instruments)

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

    providers = _build_providers(instruments)

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

            ok = failed = skipped = total_new = total_revised = 0
            for r in results:
                if r.status == "ok":
                    ok += 1
                    total_new += r.rows_new
                    total_revised += r.rows_revised
                    add_run_item(
                        session,
                        run,
                        r.instrument_id,
                        r.provider,
                        "ok",
                        rows_new=r.rows_new,
                        rows_revised=r.rows_revised,
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
                else:
                    skipped += 1

            run_status = "success" if failed == 0 else ("failed" if ok == 0 else "partial")
            finish_run(
                session,
                run,
                status=run_status,
                instruments_ok=ok,
                instruments_failed=failed,
                rows_written=total_new + total_revised,
            )
            session.commit()

    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    total_rows = total_new + total_revised
    print(
        f"backfill: {ok} ok, {failed} failed, {skipped} skipped, "
        f"{total_rows} rows ({total_new} new, {total_revised} revised)."
    )
    return 1 if (ok == 0 and failed > 0) else 0


# ── update ────────────────────────────────────────────────────────────────────


def _cmd_update(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .db.repository import (
        add_run_item,
        finish_run,
        get_last_price_date,
        start_run,
        sync_registry,
    )
    from .ingestion import PostgresPriceSink, run_backfill
    from .registry import load_registry

    try:
        instruments = load_registry()
    except Exception as exc:
        print(f"ERROR: failed to load registry: {exc}", file=sys.stderr)
        return 1

    if args.instrument:
        requested = set(args.instrument)
        instruments = [i for i in instruments if i.id in requested]
        missing = requested - {i.id for i in instruments}
        for m in sorted(missing):
            print(f"ERROR: unknown instrument ID: {m!r}", file=sys.stderr)
        if missing:
            return 1

    instruments = _active_instruments(instruments)
    providers = _build_providers(instruments)

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

            run = start_run(session, mode="update")
            session.flush()

            def get_start(instrument_id: str, provider: str) -> date | None:
                last = get_last_price_date(session, instrument_id, provider)
                if last is None:
                    return None
                return last - timedelta(days=5)

            sink = PostgresPriceSink(session)
            results = run_backfill(
                instruments,
                providers=providers,
                sink=sink,
                session=session,
                run_id=run.id,
                get_start=get_start,
            )

            ok = failed = skipped = total_new = total_revised = 0
            for r in results:
                if r.status == "ok":
                    ok += 1
                    total_new += r.rows_new
                    total_revised += r.rows_revised
                    add_run_item(
                        session,
                        run,
                        r.instrument_id,
                        r.provider,
                        "ok",
                        rows_new=r.rows_new,
                        rows_revised=r.rows_revised,
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
                    log.warning("update failed: %s — %s", r.instrument_id, r.error)
                else:
                    skipped += 1

            run_status = "success" if failed == 0 else ("failed" if ok == 0 else "partial")
            finish_run(
                session,
                run,
                status=run_status,
                instruments_ok=ok,
                instruments_failed=failed,
                rows_written=total_new + total_revised,
            )
            session.commit()

    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    total_rows = total_new + total_revised
    print(
        f"update: {ok} ok, {failed} failed, {skipped} skipped, "
        f"{total_rows} rows ({total_new} new, {total_revised} revised)."
    )
    return 1 if (ok == 0 and failed > 0) else 0


# ── status ────────────────────────────────────────────────────────────────────


def _cmd_status(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .db.repository import (
        get_flag_counts_by_rule,
        get_instrument_last_dates,
        get_last_run,
        get_price_coverage,
    )
    from .ingestion import is_stale

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            last_run = get_last_run(session)
            coverage = get_price_coverage(session)
            flag_counts = get_flag_counts_by_rule(session)
            last_dates = get_instrument_last_dates(session)
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    today = date.today()

    # ── Last run ──────────────────────────────────────────────────────────────
    print("=== Last run ===")
    if last_run is None:
        print("  No runs yet.")
    else:
        started = last_run.started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  mode:     {last_run.mode}")
        print(f"  status:   {last_run.status}")
        print(f"  started:  {started}")
        print(f"  ok:       {last_run.instruments_ok} instruments")
        print(f"  failed:   {last_run.instruments_failed} instruments")
        print(f"  rows:     {last_run.rows_written} total")

    # ── Stale instruments ─────────────────────────────────────────────────────
    stale_rows = [r for r in last_dates if is_stale(r.last_obs_date, r.native_frequency, today)]
    print(f"\n=== Stale instruments ({len(stale_rows)} of {len(last_dates)}) ===")
    if stale_rows:
        for row in stale_rows:
            last_str = str(row.last_obs_date) if row.last_obs_date else "never"
            print(f"  {row.id:<24}  {row.native_frequency:<8}  last={last_str}")
    else:
        print("  None.")

    # ── Instrument coverage ───────────────────────────────────────────────────
    print(f"\n=== Instrument coverage ({len(coverage)} series) ===")
    if coverage:
        for row in coverage:
            print(
                f"  {row.instrument_id:<24}  {row.provider:<8}  "
                f"{row.first_date}  {row.last_date}  {row.row_count} rows"
            )
    else:
        print("  No price data loaded yet.")

    # ── Data quality flags ────────────────────────────────────────────────────
    total_flags = sum(r.count for r in flag_counts)
    print(f"\n=== Data quality flags (total: {total_flags}) ===")
    if flag_counts:
        for row in flag_counts:
            print(f"  {row.rule:<32}  {row.count}")
    else:
        print("  None.")

    return 0


# ── report ────────────────────────────────────────────────────────────────────


def _cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from sqlalchemy.orm import Session

    from .coverage import build_report, write_csv, write_markdown
    from .db.engine import create_engine
    from .paths import REPO_ROOT
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
            report = build_report(session, instruments)
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    md_path = (
        Path(args.output_md)
        if args.output_md
        else REPO_ROOT / "docs" / "reports" / "symbol-coverage.md"
    )
    csv_path = (
        Path(args.output_csv)
        if args.output_csv
        else REPO_ROOT / "data" / "reports" / "symbol-coverage.csv"
    )

    try:
        write_markdown(report, md_path)
        write_csv(report, csv_path)
    except Exception as exc:
        print(f"ERROR: writing report: {exc}", file=sys.stderr)
        return 1

    mismatches = sum(1 for r in report.rows if r.fidelity_mismatch and not r.has_fidelity_override)
    print(f"report: {len(report.rows)} instruments, {mismatches} mismatch(es).")
    print(f"  {md_path}")
    print(f"  {csv_path}")

    if args.check and mismatches > 0:
        return 1
    return 0


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
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── load-ews ─────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "load-ews",
        help="Load the Bloomberg EWS CSV into the database (provider=bloomberg_ews).",
    )
    p.add_argument("--csv", type=Path, default=None, metavar="PATH")
    p.add_argument("--database-url", default=None, metavar="URL")

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
        help="Fetch data from this ISO date (YYYY-MM-DD) onward.",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── update ────────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "update",
        help="Fetch incremental data (last stored date - 5 days) for active instruments.",
    )
    p.add_argument(
        "--instrument",
        action="append",
        metavar="ID",
        help="Canonical instrument ID to update (repeatable; default: all active).",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── status ────────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "status",
        help="Show the last run summary, stale instruments, coverage, and quality flags.",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── report ────────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "report",
        help=(
            "Write symbol-coverage.md and symbol-coverage.csv from the database. "
            "Use --check to exit 1 when any YAML fidelity mismatches the computed suggestion."
        ),
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any instrument's YAML fidelity doesn't match the suggestion "
        "(unless the instrument notes contain 'fidelity-override:').",
    )
    p.add_argument(
        "--output-md",
        default=None,
        metavar="PATH",
        help="Override the markdown output path (default: docs/reports/symbol-coverage.md).",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        metavar="PATH",
        help="Override the CSV output path (default: data/reports/symbol-coverage.csv).",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    args = parser.parse_args()
    if args.command == "sync-registry":
        sys.exit(_cmd_sync_registry(args))
    elif args.command == "load-ews":
        sys.exit(_cmd_load_ews(args))
    elif args.command == "backfill":
        sys.exit(_cmd_backfill(args))
    elif args.command == "update":
        sys.exit(_cmd_update(args))
    elif args.command == "status":
        sys.exit(_cmd_status(args))
    elif args.command == "report":
        sys.exit(_cmd_report(args))
    else:
        parser.print_help()
        sys.exit(1)
