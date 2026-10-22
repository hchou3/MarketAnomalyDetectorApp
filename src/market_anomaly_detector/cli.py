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
    return [
        i
        for i in instruments
        if i.active and (not i.valid_to or date.fromisoformat(i.valid_to) >= today)
    ]


# ── sync-registry ─────────────────────────────────────────────────────────────


def _cmd_sync_registry(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .db.repository import SymbolChangeError, sync_registry
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
            try:
                changes = sync_registry(
                    session, instruments, allow_symbol_change=args.allow_symbol_change
                )
            except SymbolChangeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                print(
                    "Re-run mad-data sync-registry --allow-symbol-change to purge "
                    "stale rows and proceed.",
                    file=sys.stderr,
                )
                return 1
            session.commit()
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    n_sources = sum(len(i.sources) for i in instruments)
    print(f"sync-registry: {len(instruments)} instruments, {n_sources} sources synced.")
    for chg in changes:
        print(
            f"  symbol change: {chg.instrument_id}/{chg.provider}"
            f" {chg.old_symbol!r} → {chg.new_symbol!r}"
            f" (deleted {chg.prices_deleted} price rows, {chg.flags_deleted} flag rows)"
        )
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
                    if r.rows_revised > 0:
                        detail = f"  [{r.instrument_id}/{r.provider}] {r.rows_revised} revised"
                        if r.max_abs_revision is not None:
                            detail += f", max_abs={r.max_abs_revision:.4g}"
                        if r.max_rel_revision is not None:
                            detail += f", max_rel={r.max_rel_revision:.2%}"
                        print(detail)
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
                    if r.rows_revised > 0:
                        detail = f"  [{r.instrument_id}/{r.provider}] {r.rows_revised} revised"
                        if r.max_abs_revision is not None:
                            detail += f", max_abs={r.max_abs_revision:.4g}"
                        if r.max_rel_revision is not None:
                            detail += f", max_rel={r.max_rel_revision:.2%}"
                        print(detail)
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
        get_flag_counts_per_instrument,
        get_last_run,
        get_last_run_items,
        get_price_coverage,
        get_source_last_dates,
    )
    from .ingestion import is_stale

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            last_run = get_last_run(session)
            last_run_items = get_last_run_items(session, last_run.id) if last_run else []
            coverage = get_price_coverage(session)
            flag_counts = get_flag_counts_by_rule(session)
            flag_counts_per_inst = (
                get_flag_counts_per_instrument(session) if getattr(args, "flags", False) else []
            )
            source_dates = get_source_last_dates(session)
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

        revised_items = [it for it in last_run_items if it.rows_revised > 0]
        if revised_items:
            print(
                f"  revised:  {sum(it.rows_revised for it in revised_items)} rows across"
                f" {len(revised_items)} instrument(s)"
            )
            for it in revised_items:
                print(f"    {it.instrument_id}/{it.provider}: {it.rows_revised} revised")

    # ── Stale instruments ─────────────────────────────────────────────────────
    # Group per-source rows by instrument.
    _by_inst: dict = {}
    for row in source_dates:
        _by_inst.setdefault(row.id, []).append(row)

    stale_insts: list[tuple[str, str, str]] = []  # (id, provider, last_str)
    no_live_source_insts: list[str] = []
    for inst_id, rows in _by_inst.items():
        live = [r for r in rows if r.provider != "bloomberg_ews"]
        if not live:
            no_live_source_insts.append(inst_id)
            continue
        # Check only the priority-1 (lowest priority number) source.
        # Lower-priority fallbacks are only fetched when priority-1 fails,
        # so they are not independently stale.
        priority_1 = min(live, key=lambda r: r.priority if r.priority is not None else 999)
        freq = priority_1.source_frequency or priority_1.native_frequency
        lag = priority_1.publication_lag_days or 0
        if is_stale(priority_1.last_obs_date, freq, today, lag):
            last_str = str(priority_1.last_obs_date) if priority_1.last_obs_date else "never"
            stale_insts.append((inst_id, priority_1.provider, last_str))

    n_with_live = len(_by_inst) - len(no_live_source_insts)
    print(f"\n=== Stale instruments ({len(stale_insts)} of {n_with_live}) ===")
    if stale_insts:
        for inst_id, provider, last_str in stale_insts:
            print(f"  {inst_id:<24}  {provider:<14}  last={last_str}")
    else:
        print("  None.")

    if no_live_source_insts:
        print(f"\n=== No live source ({len(no_live_source_insts)}) ===")
        for inst_id in no_live_source_insts:
            print(f"  {inst_id}")

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

    # ── Per-instrument flag breakdown (--flags) ───────────────────────────────
    if getattr(args, "flags", False) and flag_counts_per_inst:
        print("\n=== Flag counts per instrument ===")
        for row in flag_counts_per_inst:
            print(f"  {row.instrument_id:<24}  {row.rule:<32}  {row.count}")

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


# ── timing-diagnostic ─────────────────────────────────────────────────────────

_TIMING_INSTRUMENT_IDS = [
    "gbp_usd",
    "usd_jpy",
    "gbp_usd_fred",
    "msci_japan",
    "msci_japan_n225",
]


def _cmd_revalidate(args: argparse.Namespace) -> int:
    """Re-run validation over all stored price rows; replace stale flags."""
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .db.repository import (
        delete_quality_flags_for_source,
        get_price_rows_as_df,
        write_quality_flags,
    )
    from .registry import load_instruments
    from .validation import validate_provider_output

    try:
        instruments = load_instruments()
    except Exception as exc:
        print(f"ERROR: loading registry: {exc}", file=sys.stderr)
        return 1

    try:
        engine = create_engine(url=args.database_url)
        _require_db_ready(engine)
    except SystemExit:
        return 2
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    total_deleted = 0
    total_written = 0

    with Session(engine) as session:
        with session.begin():
            for inst in instruments:
                if not getattr(inst, "active", True):
                    continue
                live_sources = [
                    s for s in inst.sources if s.provider != "bloomberg_ews" and s.symbol
                ]
                for source in live_sources:
                    df = get_price_rows_as_df(session, inst.id, source.provider)
                    if df.empty:
                        continue
                    _, flags = validate_provider_output(df, inst)
                    deleted = delete_quality_flags_for_source(session, inst.id, source.provider)
                    total_deleted += deleted
                    write_quality_flags(
                        session, flags, instrument_id=inst.id, provider=source.provider
                    )
                    total_written += len(flags)

    print(f"revalidate: cleared {total_deleted} flag(s), wrote {total_written} flag(s).")
    return 0


def _cmd_timing_diagnostic(args: argparse.Namespace) -> int:
    from pathlib import Path

    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .paths import REPO_ROOT
    from .timing import compute_timing_diagnostic, write_timing_report

    instrument_ids = args.instruments if args.instruments else _TIMING_INSTRUMENT_IDS
    out_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "docs" / "reports" / f"{args.date}-timing-diagnostic.md"
    )

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            results = compute_timing_diagnostic(session, instrument_ids)
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    try:
        write_timing_report(results, out_path)
    except Exception as exc:
        print(f"ERROR: writing report: {exc}", file=sys.stderr)
        return 1

    print(f"timing-diagnostic: {len(results)} instrument(s) analysed.")
    print(f"  {out_path}")
    return 0


# ── parity-stress ─────────────────────────────────────────────────────────────

_DEFAULT_STRESS_CSV = Path(__file__).parent.parent.parent / "data" / "raw" / "stress_extra.csv"


def _cmd_parity_stress(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .stress_parity import format_report, run_parity

    csv_path: Path = args.csv if args.csv is not None else _DEFAULT_STRESS_CSV
    if not csv_path.exists():
        print(f"ERROR: stress CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            results, exit_code, n_overlap = run_parity(session, csv_path)
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    print(format_report(results, n_overlap, csv_path))
    return exit_code


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
    p.add_argument(
        "--allow-symbol-change",
        action="store_true",
        default=False,
        help=(
            "Permit a symbol change on an existing (instrument_id, provider) pair. "
            "Atomically deletes that pair's prices and data_quality_flags rows "
            "before updating symbol_map; the next backfill refetches clean history."
        ),
    )

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
    p.add_argument(
        "--flags",
        action="store_true",
        help="Also show flag counts per instrument and rule.",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── report ────────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "report",
        help="Build the symbol coverage report (Markdown + CSV).",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any instrument has an unoverridden fidelity mismatch.",
    )
    p.add_argument("--output-md", default=None, metavar="PATH", dest="output_md")
    p.add_argument("--output-csv", default=None, metavar="PATH", dest="output_csv")
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── parity-stress ─────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "parity-stress",
        help="Verify DB stress series match stress_extra.csv on EWS Tuesdays.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to stress_extra.csv (default: data/raw/stress_extra.csv).",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── revalidate ────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "revalidate",
        help="Re-run validation over all stored price rows and replace stale flags.",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    # ── timing-diagnostic ─────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "timing-diagnostic",
        help="Compute obs_date-shift correlation table for FX/equity instruments.",
    )
    p.add_argument(
        "--instruments",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Instrument IDs to analyse (default: gbp_usd usd_jpy gbp_usd_fred"
            " msci_japan msci_japan_n225)."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output markdown path (default: docs/reports/<date>-timing-diagnostic.md).",
    )
    p.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Date prefix for the default output filename (default: today).",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    args = parser.parse_args()

    # Resolve timing-diagnostic default date now that args are parsed
    if args.command == "timing-diagnostic" and args.date is None:
        from datetime import date as _date

        args.date = str(_date.today())

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
    elif args.command == "parity-stress":
        sys.exit(_cmd_parity_stress(args))
    elif args.command == "report":
        sys.exit(_cmd_report(args))
    elif args.command == "revalidate":
        sys.exit(_cmd_revalidate(args))
    elif args.command == "timing-diagnostic":
        sys.exit(_cmd_timing_diagnostic(args))
    else:
        parser.print_help()
        sys.exit(1)
