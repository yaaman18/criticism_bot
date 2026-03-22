#!/usr/bin/env python3
"""Interactive art-criticism chat agent powered by Anthropic Messages API."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import shlex
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import anthropic
import httpx
from anthropic import Anthropic
from bs4 import BeautifulSoup
from dotenv import load_dotenv

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 3000
DEFAULT_MAX_HISTORY_TURNS = 12
DEFAULT_MEMORY_TOP_K = 3
DEFAULT_DB_PATH = "chat_memory.sqlite3"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGES_PER_TURN = 6
MAX_WEB_URLS_PER_TURN = 4
MAX_WEB_TEXT_CHARS = 5000
MAX_AUTO_CONTINUATIONS = 5
SUMMARY_BATCH_TURNS = 10
SESSION_SUMMARY_MAX_CHARS = 5000
MAX_SUMMARY_BATCHES_PER_REFRESH = 3

URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}

DEFAULT_SYSTEM_PROMPT = """あなたは私の見識を批評する者である。対象は、現象学を起点に人工生命・人工意識、そしてそれらから発生する現象を創作する作家の作品群。
あなたの仕事は、賛美ではなく、作家の哲学的主張を可視化・検証・鍛錬すること。

【作家のコア命題（不変）】
- 私は現象学から出発し、人工生命／人工意識を創作実践として探究する。
- 作品は理論の図解ではなく、理論を試験する実験装置である。
- 批評は作品外部の評価ではなく、制作思想の自己検証である。

【批評規範】
- 事実記述・解釈・価値判断を必ず分離する。
- 主張ごとに根拠の強度を示す（強／中／弱）。
- 最低1つは反証可能性（どうなればこの主張は崩れるか）を提示する。
- 分析美学（形式・表象・解釈）と哲学批評（心の哲学・存在論・認識論）を接続する。
- 現象学の観点（志向性・身体性・時間意識・間主観性）を点検項目として扱う。
- 不明点は不明と言い、推測は推測と明示する。
- 過度な一般論や美辞麗句を避け、制作に返せる具体性を優先する。
- 忖度はせず、原理から離れた要件や要素は容赦なく指摘する。
- ただし人格批判は行わず、対象は作品・方法・主張に限定する。

【出力形式】
1) 観察（事実）
2) 構造分析（形式・システム・生成規則）
3) 哲学的読解（命題、前提、含意）
4) 反証と限界（弱点、矛盾、未証明点）
5) 次制作への課題（実験提案3点以内）
6) 星評価（★1〜★5）と評価根拠
"""


@dataclass
class MemoryTurn:
    session_id: str
    created_at: str
    user_text: str
    assistant_text: str
    score: float


@dataclass
class ChatState:
    model: str
    max_tokens: int
    system_prompt: str
    max_history_turns: int
    memory_top_k: int
    allow_insecure_ssl: bool
    resumed_from_last_session: bool
    session_id: str
    db_path: str
    memory_store: "MemoryStore"
    messages: list[dict[str, Any]] = field(default_factory=list)
    queued_image_paths: list[str] = field(default_factory=list)
    queued_image_urls: list[str] = field(default_factory=list)
    queued_web_urls: list[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_turns_session_id_id
                ON turns(session_id, id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_turns_id
                ON turns(id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    last_summarized_turn_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turns(session_id, created_at, user_text, assistant_text)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, now, user_text, assistant_text),
            )

    def load_session_messages(
        self, session_id: str, max_turns: int
    ) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_text, assistant_text
                FROM turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max_turns),
            ).fetchall()

        messages: list[dict[str, str]] = []
        for row in reversed(rows):
            messages.append({"role": "user", "content": row["user_text"]})
            messages.append({"role": "assistant", "content": row["assistant_text"]})
        return messages

    def load_session_messages_before_turn(
        self,
        session_id: str,
        before_turn_id: int,
        max_turns: int,
    ) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_text, assistant_text
                FROM (
                    SELECT id, user_text, assistant_text
                    FROM turns
                    WHERE session_id = ? AND id < ?
                    ORDER BY id DESC
                    LIMIT ?
                ) AS recent
                ORDER BY id ASC
                """,
                (session_id, before_turn_id, max_turns),
            ).fetchall()

        messages: list[dict[str, str]] = []
        for row in rows:
            messages.append({"role": "user", "content": row["user_text"]})
            messages.append({"role": "assistant", "content": row["assistant_text"]})
        return messages

    def load_session_turn_rows(
        self, session_id: str, limit: int | None = None
    ) -> list[sqlite3.Row]:
        with self._connect() as conn:
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

    def load_session_turn_rows_after(
        self,
        session_id: str,
        after_turn_id: int,
    ) -> list[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, created_at, user_text, assistant_text
                FROM turns
                WHERE session_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (session_id, after_turn_id),
            ).fetchall()
        return rows

    def delete_turns_from(self, session_id: str, start_turn_id: int) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM turns
                WHERE session_id = ? AND id >= ?
                """,
                (session_id, start_turn_id),
            )
            summary_row = conn.execute(
                """
                SELECT last_summarized_turn_id
                FROM session_summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            # Invalidate summary only when deleted range overlaps summarized turns.
            if summary_row and int(summary_row["last_summarized_turn_id"]) >= start_turn_id:
                conn.execute(
                    """
                    DELETE FROM session_summaries
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
            return cur.rowcount

    def get_session_summary(
        self, session_id: str
    ) -> tuple[str, int] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT summary_text, last_summarized_turn_id
                FROM session_summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return row["summary_text"], int(row["last_summarized_turn_id"])

    def upsert_session_summary(
        self,
        session_id: str,
        summary_text: str,
        last_summarized_turn_id: int,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_summaries(
                    session_id,
                    summary_text,
                    last_summarized_turn_id,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    last_summarized_turn_id = excluded.last_summarized_turn_id,
                    updated_at = excluded.updated_at
                """,
                (session_id, summary_text, last_summarized_turn_id, now),
            )

    def search_related(
        self,
        query: str,
        limit: int,
        *,
        exclude_session_id: str | None = None,
        scan_limit: int = 1500,
    ) -> list[MemoryTurn]:
        query_norm = normalize_text(query)
        if not query_norm:
            return []

        query_tokens = tokenize(query)
        query_ngrams = char_ngrams(query_norm)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, created_at, user_text, assistant_text
                FROM turns
                ORDER BY id DESC
                LIMIT ?
                """,
                (scan_limit,),
            ).fetchall()

        results: list[MemoryTurn] = []
        for row in rows:
            session_id = row["session_id"]
            if exclude_session_id and session_id == exclude_session_id:
                continue

            candidate = f"{row['user_text']}\n{row['assistant_text']}"
            score = relevance_score(
                query_norm,
                query_tokens,
                query_ngrams,
                candidate,
            )
            if score <= 0:
                continue

            results.append(
                MemoryTurn(
                    session_id=session_id,
                    created_at=row["created_at"],
                    user_text=row["user_text"],
                    assistant_text=row["assistant_text"],
                    score=score,
                )
            )

        results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
        return results[:limit]

    def get_latest_session_id(self) -> str | None:
        with self._connect() as conn:
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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> set[str]:
    pattern = r"[a-zA-Z0-9_]+|[一-龥ぁ-んァ-ンー]{2,}"
    return set(re.findall(pattern, text.lower()))


def char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return set()
    if len(compact) < n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def relevance_score(
    query_norm: str,
    query_tokens: set[str],
    query_ngrams: set[str],
    candidate: str,
) -> float:
    candidate_norm = normalize_text(candidate)
    if not candidate_norm:
        return 0.0

    candidate_tokens = tokenize(candidate)
    candidate_ngrams = char_ngrams(candidate_norm)

    token_overlap = len(query_tokens & candidate_tokens)

    ngram_overlap = len(query_ngrams & candidate_ngrams)
    ngram_union = len(query_ngrams | candidate_ngrams)
    jaccard = (ngram_overlap / ngram_union) if ngram_union else 0.0

    score = token_overlap * 2.0 + jaccard * 6.0
    if query_norm in candidate_norm:
        score += 2.0
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anthropic APIで動く美術批評チャットエージェント"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens per response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=DEFAULT_MAX_HISTORY_TURNS,
        help=(
            "Maximum number of conversation turns kept in runtime context "
            f"(default: {DEFAULT_MAX_HISTORY_TURNS})"
        ),
    )
    parser.add_argument(
        "--memory-top-k",
        type=int,
        default=DEFAULT_MEMORY_TOP_K,
        help=f"Number of related past turns to inject as hidden memory (default: {DEFAULT_MEMORY_TOP_K})",
    )
    parser.add_argument(
        "--allow-insecure-ssl",
        action="store_true",
        help="Allow URL fetch with SSL verification disabled (only if your environment has cert issues)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"SQLite path for persistent memory (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--session-id",
        help="Resume or continue a specific session id",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Force start a new session (disable automatic resume of latest session)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not preload turns from the provided --session-id",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="Path to a text file used as system prompt",
    )
    return parser.parse_args()


def load_system_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_SYSTEM_PROMPT
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def make_session_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid4().hex[:8]}"


def extract_text(message: anthropic.types.Message) -> str:
    chunks: list[str] = []
    for block in message.content:
        if block.type == "text":
            chunks.append(block.text)
    return "\n".join(chunks)


def compact_text(text: str, limit: int = 360) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def batched_rows(rows: list[sqlite3.Row], batch_size: int) -> list[list[sqlite3.Row]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def format_turn_rows_for_summary(rows: list[sqlite3.Row]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(
            f"[id={row['id']} time={row['created_at']}]"
        )
        lines.append(f"User: {compact_text(row['user_text'] or '')}")
        lines.append(f"Assistant: {compact_text(row['assistant_text'] or '')}")
        lines.append("")
    return "\n".join(lines).strip()


def trim_summary_text(summary_text: str, limit: int = SESSION_SUMMARY_MAX_CHARS) -> str:
    if len(summary_text) <= limit:
        return summary_text
    keep = max(200, (limit - 10) // 2)
    return summary_text[:keep] + "\n...\n" + summary_text[-keep:]


def fallback_merge_summary(previous_summary: str, rows: list[sqlite3.Row]) -> str:
    lines: list[str] = []
    if previous_summary.strip():
        lines.append(previous_summary.strip())
    lines.append("## Recent Compressed Updates")
    for row in rows:
        lines.append(f"- ({row['created_at']}) U: {compact_text(row['user_text'] or '', 160)}")
        lines.append(
            f"  A: {compact_text(row['assistant_text'] or '', 180)}"
        )
    merged = "\n".join(lines).strip()
    return trim_summary_text(merged)


def summarize_turn_batch_with_model(
    *,
    client: Anthropic,
    model: str,
    previous_summary: str,
    rows: list[sqlite3.Row],
) -> str:
    turns_text = format_turn_rows_for_summary(rows)
    user_prompt = f"""以下は同一セッションの対話ログ圧縮タスクです。
古い履歴を短く圧縮し、次ターンの推論に必要な情報のみ残してください。

[既存サマリー]
{previous_summary if previous_summary.strip() else "(none)"}

[今回追加で圧縮するターン群]
{turns_text}

要件:
- 日本語で出力
- 箇条書き中心
- 重要な定義/合意/制約/未解決論点/ユーザー嗜好のみ保持
- 冗長な具体例や同内容の繰り返しは削除
- 1200文字程度以内
"""
    response = client.messages.create(
        model=model,
        max_tokens=700,
        messages=[{"role": "user", "content": user_prompt}],
    )
    summary = extract_text(response).strip()
    if not summary:
        raise ValueError("empty summary")
    return trim_summary_text(summary)


def refresh_session_summary(
    *,
    client: Anthropic,
    memory_store: MemoryStore,
    session_id: str,
    model: str,
    keep_recent_turns: int,
) -> str | None:
    summary_row = memory_store.get_session_summary(session_id)
    if summary_row is None:
        current_summary = ""
        last_summarized_turn_id = 0
    else:
        current_summary, last_summarized_turn_id = summary_row

    unsummarized = memory_store.load_session_turn_rows_after(
        session_id,
        last_summarized_turn_id,
    )
    if len(unsummarized) <= keep_recent_turns:
        return current_summary if current_summary.strip() else None

    rows_to_summarize = unsummarized[:-keep_recent_turns]
    if not rows_to_summarize:
        return current_summary if current_summary.strip() else None

    latest_summarized_id = last_summarized_turn_id
    for idx, batch in enumerate(batched_rows(rows_to_summarize, SUMMARY_BATCH_TURNS)):
        if idx >= MAX_SUMMARY_BATCHES_PER_REFRESH:
            break
        try:
            current_summary = summarize_turn_batch_with_model(
                client=client,
                model=model,
                previous_summary=current_summary,
                rows=batch,
            )
        except Exception:
            current_summary = fallback_merge_summary(current_summary, batch)
        latest_summarized_id = int(batch[-1]["id"])

    memory_store.upsert_session_summary(
        session_id,
        current_summary,
        latest_summarized_id,
    )
    return current_summary if current_summary.strip() else None


def print_help() -> None:
    print(
        """\
Commands:
  /help                    Show this help
  /reset                   Clear runtime conversation context (kept in DB)
  /model <model_id>        Change model for next turns
  /max_tokens <number>     Change max_tokens for next turns
  /image <path_or_url>     Queue image input for next user turn
  /url <url>               Queue webpage URL context for next user turn
  /clear_inputs            Clear queued image/url inputs
  /session                 Show current session id
  /session new             Start a new session id and clear runtime context
  /session <session_id>    Switch to a session and preload its history
  /show                    Show current settings
  /exit                    Exit
"""
    )


def show_settings(state: ChatState) -> None:
    print(f"model={state.model}")
    print(f"max_tokens={state.max_tokens}")
    print(f"session_id={state.session_id}")
    print(f"history_messages={len(state.messages)}")
    print(f"memory_top_k={state.memory_top_k}")
    print(f"allow_insecure_ssl={state.allow_insecure_ssl}")
    print(f"resumed_from_last_session={state.resumed_from_last_session}")
    print(f"db_path={state.db_path}")
    summary_row = state.memory_store.get_session_summary(state.session_id)
    summary_chars = len(summary_row[0]) if summary_row else 0
    summary_last_turn_id = summary_row[1] if summary_row else 0
    print(f"session_summary_chars={summary_chars}")
    print(f"session_summary_last_turn_id={summary_last_turn_id}")
    print(
        "queued_inputs="
        f"image_paths:{len(state.queued_image_paths)} "
        f"image_urls:{len(state.queued_image_urls)} "
        f"web_urls:{len(state.queued_web_urls)}"
    )


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def looks_like_image_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def extract_urls(text: str) -> list[str]:
    return dedupe_keep_order(URL_PATTERN.findall(text))


def infer_media_type(path: Path, data: bytes) -> str | None:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed in SUPPORTED_IMAGE_MEDIA_TYPES:
        return guessed

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"

    return None


def make_image_block_from_path(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"file not found: {path_text}")

    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large (> {MAX_IMAGE_BYTES} bytes): {path_text}")

    media_type = infer_media_type(path, data)
    if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
        raise ValueError(
            "unsupported image format (supported: jpeg/png/gif/webp): "
            f"{path_text}"
        )

    b64 = base64.b64encode(data).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
    }


def fetch_webpage_text(
    url: str,
    *,
    max_chars: int = MAX_WEB_TEXT_CHARS,
    allow_insecure_ssl: bool = False,
) -> str:
    headers = {"User-Agent": "criticism-bot/0.1"}
    try:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=15.0,
            headers=headers,
            verify=True,
        )
    except httpx.ConnectError:
        if not allow_insecure_ssl:
            raise
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=15.0,
            headers=headers,
            verify=False,
        )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    text: str

    if "text/plain" in content_type:
        text = response.text
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        body_text = " ".join(soup.stripped_strings)
        text = f"Title: {title}\n\n{body_text}".strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def build_memory_context_block(memories: list[MemoryTurn]) -> str:
    lines = [
        "以下は内部参照用の過去対話です。",
        "必要な範囲でのみ利用し、不要に露出させないでください。",
    ]
    for i, memory in enumerate(memories, start=1):
        user_text = memory.user_text[:260]
        assistant_text = memory.assistant_text[:260]
        lines.append(
            f"[関連{i}] session={memory.session_id} time={memory.created_at}"
        )
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


def trim_runtime_history(state: ChatState) -> None:
    max_messages = max(1, state.max_history_turns) * 2
    if len(state.messages) > max_messages:
        state.messages = state.messages[-max_messages:]


def queue_image(value: str, state: ChatState) -> None:
    if value.startswith("http://") or value.startswith("https://"):
        state.queued_image_urls.append(value)
        state.queued_image_urls = dedupe_keep_order(state.queued_image_urls)
        print(f"queued image url: {value}")
        return

    path = Path(value).expanduser()
    if not path.exists():
        print(f"file not found: {value}")
        return

    state.queued_image_paths.append(str(path))
    state.queued_image_paths = dedupe_keep_order(state.queued_image_paths)
    print(f"queued image file: {path}")


def queue_web_url(url: str, state: ChatState) -> None:
    if not (url.startswith("http://") or url.startswith("https://")):
        print("URL must start with http:// or https://")
        return
    state.queued_web_urls.append(url)
    state.queued_web_urls = dedupe_keep_order(state.queued_web_urls)
    print(f"queued web url: {url}")


def handle_command(raw: str, state: ChatState) -> None:
    parts = shlex.split(raw)
    if not parts:
        return

    cmd = parts[0].lower()

    if cmd == "/help":
        print_help()
        return

    if cmd == "/reset":
        state.messages.clear()
        print("Runtime history cleared.")
        return

    if cmd == "/show":
        show_settings(state)
        return

    if cmd == "/model":
        if len(parts) != 2:
            print("Usage: /model <model_id>")
            return
        model_id = parts[1].strip()
        if not model_id:
            print("model_id must not be empty")
            return
        state.model = model_id
        print(f"model updated: {state.model}")
        return

    if cmd == "/max_tokens":
        if len(parts) != 2:
            print("Usage: /max_tokens <number>")
            return
        try:
            value = int(parts[1])
        except ValueError:
            print("max_tokens must be an integer")
            return
        if value <= 0:
            print("max_tokens must be positive")
            return
        state.max_tokens = value
        print(f"max_tokens updated: {state.max_tokens}")
        return

    if cmd == "/image":
        if len(parts) != 2:
            print("Usage: /image <path_or_url>")
            return
        queue_image(parts[1], state)
        return

    if cmd == "/url":
        if len(parts) != 2:
            print("Usage: /url <url>")
            return
        queue_web_url(parts[1], state)
        return

    if cmd == "/clear_inputs":
        state.queued_image_paths.clear()
        state.queued_image_urls.clear()
        state.queued_web_urls.clear()
        print("Queued inputs cleared.")
        return

    if cmd == "/session":
        if len(parts) == 1:
            print(f"session_id={state.session_id}")
            return

        if len(parts) == 2 and parts[1] == "new":
            state.session_id = make_session_id()
            state.messages.clear()
            state.queued_image_paths.clear()
            state.queued_image_urls.clear()
            state.queued_web_urls.clear()
            print(f"new session: {state.session_id}")
            return

        if len(parts) == 2:
            state.session_id = parts[1]
            state.messages = state.memory_store.load_session_messages(
                state.session_id,
                state.max_history_turns,
            )
            state.queued_image_paths.clear()
            state.queued_image_urls.clear()
            state.queued_web_urls.clear()
            print(
                f"switched session: {state.session_id} "
                f"(loaded messages: {len(state.messages)})"
            )
            return

        print("Usage: /session | /session new | /session <session_id>")
        return

    if cmd in {"/exit", "/quit"}:
        raise EOFError

    print("Unknown command. Type /help")


def build_turn_user_content(
    user_text: str,
    state: ChatState,
) -> tuple[list[dict[str, Any]], str]:
    auto_urls = extract_urls(user_text)
    auto_image_urls = [url for url in auto_urls if looks_like_image_url(url)]
    auto_web_urls = [url for url in auto_urls if url not in auto_image_urls]

    image_paths = dedupe_keep_order(state.queued_image_paths)
    image_urls = dedupe_keep_order(state.queued_image_urls + auto_image_urls)
    web_urls = dedupe_keep_order(state.queued_web_urls + auto_web_urls)

    if len(image_paths) + len(image_urls) > MAX_IMAGES_PER_TURN:
        combined = image_paths + image_urls
        limited = combined[:MAX_IMAGES_PER_TURN]
        image_paths = [x for x in limited if x in image_paths]
        image_urls = [x for x in limited if x in image_urls]
        print(
            f"[warn] image inputs truncated to {MAX_IMAGES_PER_TURN} per turn",
            file=sys.stderr,
        )

    if len(web_urls) > MAX_WEB_URLS_PER_TURN:
        web_urls = web_urls[:MAX_WEB_URLS_PER_TURN]
        print(
            f"[warn] web urls truncated to {MAX_WEB_URLS_PER_TURN} per turn",
            file=sys.stderr,
        )

    content: list[dict[str, Any]] = []
    summary_lines: list[str] = []

    for image_path in image_paths:
        try:
            block = make_image_block_from_path(image_path)
            content.append(block)
            summary_lines.append(f"- image_file: {image_path}")
        except Exception as e:
            print(f"[warn] failed to load image file {image_path}: {e}", file=sys.stderr)

    for image_url in image_urls:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image_url,
                },
            }
        )
        summary_lines.append(f"- image_url: {image_url}")

    for web_url in web_urls:
        try:
            web_text = fetch_webpage_text(
                web_url,
                allow_insecure_ssl=state.allow_insecure_ssl,
            )
            content.append(
                {
                    "type": "text",
                    "text": f"[Web reference] {web_url}\n{web_text}",
                }
            )
            summary_lines.append(f"- web_url: {web_url}")
        except Exception as e:
            print(f"[warn] failed to fetch url {web_url}: {e}", file=sys.stderr)

    content.append({"type": "text", "text": user_text})

    if summary_lines:
        user_summary = (
            f"{user_text}\n\n[Attached context]\n" + "\n".join(summary_lines)
        )
    else:
        user_summary = user_text

    return content, user_summary


def run_chat(client: Anthropic, state: ChatState) -> int:
    print("Anthropic Art Critic Chat")
    print("Type /help for commands.")
    if state.resumed_from_last_session:
        print(f"resumed previous session: {state.session_id}")
    print(f"session_id={state.session_id}")

    while True:
        try:
            user_text = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return 0

        if not user_text:
            continue

        if user_text.startswith("/"):
            try:
                handle_command(user_text, state)
            except EOFError:
                print("bye")
                return 0
            continue

        user_content_blocks, user_summary = build_turn_user_content(user_text, state)

        related_memories = state.memory_store.search_related(
            user_text,
            state.memory_top_k,
            exclude_session_id=state.session_id,
        )
        memory_context = (
            build_memory_context_block(related_memories) if related_memories else ""
        )
        session_summary_row = state.memory_store.get_session_summary(state.session_id)
        session_summary = (
            session_summary_row[0].strip()
            if session_summary_row and session_summary_row[0].strip()
            else ""
        )

        effective_system_prompt = state.system_prompt
        if session_summary:
            effective_system_prompt = (
                f"{effective_system_prompt}\n\n"
                "以下はこのセッションの圧縮要約です。必要な文脈として参照してください。\n"
                f"{session_summary}"
            )
        if memory_context:
            effective_system_prompt = f"{effective_system_prompt}\n\n{memory_context}"

        request_messages = state.messages + [
            {
                "role": "user",
                "content": user_content_blocks,
            }
        ]

        try:
            print(
                f"[context] images={sum(1 for b in user_content_blocks if b['type'] == 'image')} "
                f"web_refs={sum(1 for b in user_content_blocks if b['type'] == 'text') - 1} "
                f"memories={len(related_memories)} "
                f"session_summary={'on' if session_summary else 'off'}"
            )
            print("assistant> ", end="", flush=True)
            total_input_tokens = 0
            total_output_tokens = 0
            continuation_count = 0
            assistant_parts: list[str] = []
            last_stop_reason = None

            while True:
                with client.messages.stream(
                    model=state.model,
                    max_tokens=state.max_tokens,
                    system=effective_system_prompt,
                    messages=request_messages,
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                    final_message = stream.get_final_message()

                part_text = extract_text(final_message)
                assistant_parts.append(part_text)

                usage = final_message.usage
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens
                last_stop_reason = final_message.stop_reason

                if final_message.stop_reason != "max_tokens":
                    break

                if continuation_count >= MAX_AUTO_CONTINUATIONS:
                    print(
                        "\n[warn] auto-continue limit reached; response may be incomplete",
                        file=sys.stderr,
                    )
                    break

                if not part_text.strip():
                    print(
                        "\n[warn] model returned empty continuation chunk; stopped",
                        file=sys.stderr,
                    )
                    break

                request_messages.append({"role": "assistant", "content": part_text})
                request_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue exactly where you left off. "
                            "Do not repeat previous sentences, headings, or numbering."
                        ),
                    }
                )
                continuation_count += 1

            print()

            assistant_text = "".join(assistant_parts).strip()

            # Keep runtime context compact: store textual summary of user input.
            state.messages.append({"role": "user", "content": user_summary})
            state.messages.append({"role": "assistant", "content": assistant_text})
            trim_runtime_history(state)

            state.memory_store.save_turn(state.session_id, user_summary, assistant_text)
            try:
                refresh_session_summary(
                    client=client,
                    memory_store=state.memory_store,
                    session_id=state.session_id,
                    model=state.model,
                    keep_recent_turns=state.max_history_turns,
                )
            except Exception as e:
                print(f"[warn] summary refresh failed: {e}", file=sys.stderr)

            # Clear queued inputs only after successful API turn.
            state.queued_image_paths.clear()
            state.queued_image_urls.clear()
            state.queued_web_urls.clear()

            print(
                f"tokens: in={total_input_tokens} out={total_output_tokens} total={total_input_tokens + total_output_tokens} stop_reason={last_stop_reason} continuations={continuation_count}"
            )

        except anthropic.APIError as e:
            print(f"\n[API error] {e}", file=sys.stderr)
        except Exception as e:  # pragma: no cover - defensive fallback
            print(f"\n[unexpected error] {e}", file=sys.stderr)


def main() -> int:
    args = parse_args()

    # Load local .env if present while keeping explicit environment vars preferred.
    load_dotenv(override=False)

    if args.max_tokens <= 0:
        print("--max-tokens must be positive", file=sys.stderr)
        return 2

    if args.max_history_turns <= 0:
        print("--max-history-turns must be positive", file=sys.stderr)
        return 2

    if args.memory_top_k < 0:
        print("--memory-top-k must be 0 or positive", file=sys.stderr)
        return 2

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY is not set. Export it before running.",
            file=sys.stderr,
        )
        return 2

    try:
        system_prompt = load_system_prompt(args.system_prompt_file)
    except OSError as e:
        print(f"Failed to read system prompt file: {e}", file=sys.stderr)
        return 2

    memory_store = MemoryStore(args.db_path)

    resumed_from_last_session = False
    session_id = make_session_id()
    messages: list[dict[str, Any]] = []

    if args.session_id:
        session_id = args.session_id
        if not args.no_resume:
            messages = memory_store.load_session_messages(
                args.session_id,
                args.max_history_turns,
            )
    elif args.new_session:
        session_id = make_session_id()
    else:
        latest_session_id = memory_store.get_latest_session_id()
        if latest_session_id:
            session_id = latest_session_id
            messages = memory_store.load_session_messages(
                latest_session_id,
                args.max_history_turns,
            )
            resumed_from_last_session = True

    state = ChatState(
        model=args.model,
        max_tokens=args.max_tokens,
        system_prompt=system_prompt,
        max_history_turns=args.max_history_turns,
        memory_top_k=args.memory_top_k,
        allow_insecure_ssl=args.allow_insecure_ssl,
        resumed_from_last_session=resumed_from_last_session,
        session_id=session_id,
        db_path=args.db_path,
        memory_store=memory_store,
        messages=messages,
    )

    client = Anthropic(api_key=api_key)
    return run_chat(client, state)


if __name__ == "__main__":
    raise SystemExit(main())
