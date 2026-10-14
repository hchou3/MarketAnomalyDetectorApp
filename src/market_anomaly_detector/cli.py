"""mad-data CLI — ingestion and data-management commands."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _cmd_load_ews(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .providers.ews import load_ews

    engine = create_engine(url=args.database_url)
    with Session(engine) as session:
        result = load_ews(session, csv_path=args.csv)
        session.commit()

    print(
        f"load-ews: {result['prices_written']} price rows, "
        f"{result['labels_written']} label rows, "
        f"{result['instruments_ok']} ok, "
        f"{result['instruments_failed']} failed."
    )
    return 1 if result["instruments_failed"] > 0 else 0


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="mad-data",
        description="Market Anomaly Detector — data management commands.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

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
        help="Database URL (default: MAD_DATABASE_URL env var or the built-in default).",
    )

    args = parser.parse_args()
    if args.command == "load-ews":
        sys.exit(_cmd_load_ews(args))
    else:
        parser.print_help()
        sys.exit(1)
