"""mad-model CLI — model management commands."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_REPORTS_DIR = Path(__file__).parent.parent.parent / "docs" / "reports"

_DB_NOT_INIT_MSG = "Database not initialized — run: uv run alembic upgrade head"

# Known historical crisis windows used to annotate episodes in the report.
# Each entry: (start_iso, end_iso, label).
_KNOWN_CRISES: list[tuple[str, str, str]] = [
    ("2007-06-01", "2007-12-31", "2007 subprime onset"),
    ("2008-01-01", "2009-06-30", "2008 financial crisis"),
    ("2010-04-01", "2010-09-30", "2010 Eurozone / flash crash"),
    ("2011-06-01", "2012-06-30", "2011–12 Euro debt crisis"),
    ("2013-05-01", "2013-08-31", "2013 taper tantrum"),
    ("2014-09-01", "2015-02-28", "2014–15 oil shock"),
    ("2015-06-01", "2016-04-30", "2015–16 China slowdown / oil crash"),
    ("2018-10-01", "2019-02-28", "2018 Q4 global equity selloff"),
    ("2020-01-01", "2020-12-31", "2020 COVID-19"),
]


def _historical_match(start: date, end: date) -> str:
    """Return the best-matching known crisis label for an episode, or empty string."""
    best_label = ""
    best_overlap = 0
    ep_start = start.toordinal()
    ep_end = end.toordinal()
    for cs, ce, label in _KNOWN_CRISES:
        cs_ord = date.fromisoformat(cs).toordinal()
        ce_ord = date.fromisoformat(ce).toordinal()
        overlap = max(0, min(ep_end, ce_ord) - max(ep_start, cs_ord))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def _require_db_ready(engine) -> None:
    from sqlalchemy import inspect as sa_inspect

    insp = sa_inspect(engine)
    if not insp.has_table("instruments"):
        print(_DB_NOT_INIT_MSG, file=sys.stderr)
        raise SystemExit(2)


# ── episodes ──────────────────────────────────────────────────────────────────


def _cmd_episodes(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from .db.engine import create_engine
    from .episodes import extract_episodes
    from .labels import load_labels

    label_set: str = args.label_set
    merge_gap: int = args.merge_gap

    try:
        engine = create_engine(url=args.database_url)
        with Session(engine) as session:
            _require_db_ready(engine)
            labels = load_labels(session, label_set=label_set)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: database error: {exc}", file=sys.stderr)
        return 1

    if labels.empty:
        print(f"ERROR: no labels found for label_set={label_set!r}", file=sys.stderr)
        return 1

    episodes_by_gap = {g: extract_episodes(labels, merge_gap=g) for g in (0, 2, 4)}
    main_episodes = extract_episodes(labels, merge_gap=merge_gap)

    out_path = _REPORTS_DIR / f"episodes-{label_set}.md"
    _write_report(out_path, label_set, merge_gap, main_episodes, labels, episodes_by_gap)

    print(
        f"episodes: {len(main_episodes)} episodes (merge_gap={merge_gap})"
        f" → {out_path.relative_to(Path(__file__).parent.parent.parent)}"
    )
    return 0


def _write_report(
    out_path: Path,
    label_set: str,
    merge_gap: int,
    episodes,
    labels,
    episodes_by_gap: dict[int, list],
) -> None:
    today = date.today().isoformat()
    label_count = int((labels["value"] == 1).sum())
    total_weeks = len(labels)
    date_range = f"{labels['week'].iloc[0]} – {labels['week'].iloc[-1]}"

    lines: list[str] = [
        f"# Anomaly Episodes — {label_set}",
        "",
        f"Generated: {today}  ",
        f"Label set: `{label_set}`  ",
        f"Date range: {date_range}  ",
        f"Total weeks: {total_weeks}  ",
        f"Anomaly weeks: {label_count} ({100 * label_count / total_weeks:.1f}%)",
        "",
        f"## Episode list (merge gap = {merge_gap} weeks)",
        "",
        "| # | Start | End | Weeks | Historical match |",
        "|---|-------|-----|-------|-----------------|",
    ]

    for i, ep in enumerate(episodes, 1):
        match = _historical_match(ep.start, ep.end)
        lines.append(f"| {i} | {ep.start} | {ep.end} | {ep.length} | {match} |")

    lines += [
        "",
        f"**Total: {len(episodes)} episode(s)**",
        "",
        "## Episode count by merge gap",
        "",
        "| Merge gap (calm weeks) | Episodes |",
        "|------------------------|----------|",
    ]
    for g in (0, 2, 4):
        lines.append(f"| {g} | {len(episodes_by_gap[g])} |")

    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mad-model",
        description="Market Anomaly Detector — model management commands.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── episodes ─────────────────────────────────────────────────────────────
    p = subparsers.add_parser(
        "episodes",
        help=(
            "Extract anomaly episodes from the labels table and write "
            "docs/reports/episodes-<label_set>.md."
        ),
    )
    p.add_argument(
        "--label-set",
        default="ews_v1",
        metavar="NAME",
        help="Label set to read from the database (default: ews_v1).",
    )
    p.add_argument(
        "--merge-gap",
        type=int,
        default=2,
        metavar="N",
        help="Maximum calm weeks between runs to merge into one episode (default: 2).",
    )
    p.add_argument("--database-url", default=None, metavar="URL")

    args = parser.parse_args()
    if args.command == "episodes":
        sys.exit(_cmd_episodes(args))
    else:
        parser.print_help()
        sys.exit(1)
