#!/usr/bin/env python3
"""Export chat logs stored in SQLite to a readable Markdown file."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export criticism_bot SQLite chat logs to Markdown"
    )
    parser.add_argument(
        "--db-path",
        default="chat_memory.sqlite3",
        help="Path to SQLite DB (default: chat_memory.sqlite3)",
    )
    parser.add_argument(
        "--session-id",
        help="Export this session id. If omitted, latest session is used.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of turns (from newest), then output in chronological order.",
    )
    parser.add_argument(
        "--output",
        help="Output Markdown path. If omitted, auto-generated under exports/.",
    )
    return parser.parse_args()


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_session_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT session_id
        FROM turns
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return row["session_id"]


def fetch_turns(
    conn: sqlite3.Connection, session_id: str, limit: int | None
) -> list[sqlite3.Row]:
    if limit is None:
        rows = conn.execute(
            """
            SELECT id, session_id, created_at, user_text, assistant_text
            FROM turns
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return rows

    newest_rows = conn.execute(
        """
        SELECT id, session_id, created_at, user_text, assistant_text
        FROM turns
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    return list(reversed(newest_rows))


def code_block(text: str) -> str:
    fence = "```"
    if "```" in text:
        fence = "````"
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}text\n{body}{fence}\n"


def build_markdown(
    *, db_path: str, session_id: str, turns: list[sqlite3.Row]
) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# Chat Log Export")
    lines.append("")
    lines.append(f"- Generated at: `{now}`")
    lines.append(f"- Database: `{db_path}`")
    lines.append(f"- Session: `{session_id}`")
    lines.append(f"- Turns: `{len(turns)}`")
    lines.append("")

    for idx, row in enumerate(turns, start=1):
        lines.append(f"## Turn {idx}")
        lines.append("")
        lines.append(f"- id: `{row['id']}`")
        lines.append(f"- created_at: `{row['created_at']}`")
        lines.append("")
        lines.append("### User")
        lines.append("")
        lines.append(code_block(row["user_text"] or ""))
        lines.append("### Assistant")
        lines.append("")
        lines.append(code_block(row["assistant_text"] or ""))

    return "\n".join(lines).strip() + "\n"


def ensure_output_path(args_output: str | None, session_id: str) -> Path:
    if args_output:
        path = Path(args_output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exports_dir = Path("exports")
    exports_dir.mkdir(parents=True, exist_ok=True)
    return exports_dir / f"chat_log_{session_id}_{stamp}.md"


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).expanduser()
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 2

    conn = get_connection(str(db_path))
    try:
        session_id = args.session_id or get_latest_session_id(conn)
        if session_id is None:
            print("No logs found in DB.")
            return 1

        if args.limit is not None and args.limit <= 0:
            print("--limit must be positive")
            return 2

        turns = fetch_turns(conn, session_id, args.limit)
        if not turns:
            print(f"No turns found for session: {session_id}")
            return 1
    finally:
        conn.close()

    output_path = ensure_output_path(args.output, session_id)
    md = build_markdown(db_path=str(db_path), session_id=session_id, turns=turns)
    output_path.write_text(md, encoding="utf-8")

    print(f"Exported: {output_path}")
    print(f"Session: {session_id}")
    print(f"Turns: {len(turns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
