#!/usr/bin/env python3
"""Web UI for the criticism bot using Streamlit."""

from __future__ import annotations

import copy
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import anthropic
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from anthropic_art_critic_chat import (
    DEFAULT_MAX_HISTORY_TURNS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEMORY_TOP_K,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    MAX_AUTO_CONTINUATIONS,
    MAX_IMAGES_PER_TURN,
    MAX_WEB_URLS_PER_TURN,
    MemoryStore,
    build_memory_context_block,
    dedupe_keep_order,
    extract_text,
    extract_urls,
    fetch_webpage_text,
    infer_media_type,
    looks_like_image_url,
    make_session_id,
    refresh_session_summary,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(BASE_DIR / "chat_memory.sqlite3")
SUPPORTED_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_TEXT_FILES_PER_TURN = 5
MAX_TEXT_FILE_BYTES = 350_000
MAX_TEXT_FILE_CHARS = 12_000
DEFAULT_MAX_INPUT_TOKENS = 180_000
MAX_INPUT_TOKENS_UI_MAX = 400_000
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 3.2
TOKEN_ESTIMATE_IMAGE_TOKENS = 1200
MIN_ATTACHMENT_TEXT_CHARS = 900
MIN_USER_TEXT_CHARS = 350
ATTACHMENT_TEXT_PREFIXES = ("[Web reference]", "[File reference]")
TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".log",
    ".ini",
    ".toml",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sql",
    ".sh",
}


def trim_runtime_messages(
    messages: list[dict[str, Any]], max_history_turns: int
) -> list[dict[str, Any]]:
    max_messages = max(1, int(max_history_turns)) * 2
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def apply_styles(theme_mode: str) -> None:
    if theme_mode == "Light":
        theme_vars = """
        :root {
          --bg: #f6f7f8;
          --panel: #ffffff;
          --card: #ffffff;
          --ink: #111827;
          --muted: #4b5563;
          --line: #d1d5db;
          --accent: #2563eb;
        }
        """
    else:
        theme_vars = """
        :root {
          --bg: #07090d;
          --panel: #0d1118;
          --card: #111827;
          --ink: #ffffff;
          --muted: #c7ced9;
          --line: #273244;
          --accent: #6ea8ff;
        }
        """

    style_template = """
        <style>
        __THEME_VARS__
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child,
        .stApp {
          background: var(--bg) !important;
        }
        .block-container {
          max-width: 1100px;
          padding-top: 1.1rem;
          background: var(--bg) !important;
        }
        .stApp * {
          color: var(--ink) !important;
        }
        a {
          color: var(--accent) !important;
        }
        .meta {
          color: var(--muted) !important;
          font-size: 0.92rem;
        }
        .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
          color: var(--ink) !important;
        }
        .stChatMessage {
          border: 1px solid var(--line) !important;
          border-radius: 14px;
          padding: 0.55rem 0.75rem;
          background: var(--card) !important;
        }
        .stChatMessage[data-testid="chat-message-user"] {
          border-left: 4px solid #60a5fa !important;
        }
        .stChatMessage[data-testid="chat-message-assistant"] {
          border-left: 4px solid #22d3ee !important;
        }
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input,
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox [data-baseweb="select"] > div,
        .stFileUploader section,
        .stFileUploader > div {
          background: var(--panel) !important;
          color: var(--ink) !important;
          border-color: var(--line) !important;
        }
        input::placeholder, textarea::placeholder {
          color: var(--muted) !important;
          opacity: 1 !important;
        }
        .stButton > button,
        .stDownloadButton > button {
          background: var(--panel) !important;
          color: var(--ink) !important;
          border: 1px solid var(--line) !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
          border-color: #4b5e7c !important;
        }
        pre, code, kbd, samp {
          background: var(--panel) !important;
          color: var(--ink) !important;
          border: 1px solid var(--line) !important;
        }
        hr {
          border-color: var(--line) !important;
        }
        </style>
        """
    st.markdown(
        style_template.replace("__THEME_VARS__", theme_vars),
        unsafe_allow_html=True,
    )


def get_recent_sessions(db_path: str, limit: int = 200) -> list[str]:
    path = Path(db_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT session_id, MAX(id) AS latest_id
            FROM turns
            GROUP BY session_id
            ORDER BY latest_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [row["session_id"] for row in rows]


def runtime_to_display_messages(runtime_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    display: list[dict[str, str]] = []
    for message in runtime_messages:
        role = message.get("role", "assistant")
        content = str(message.get("content", ""))
        display.append({"role": role, "content": content})
    return display


def parse_url_input(raw: str) -> list[str]:
    if not raw.strip():
        return []
    parts = re.split(r"[\n,\s]+", raw.strip())
    urls = [p for p in parts if p.startswith("http://") or p.startswith("https://")]
    return dedupe_keep_order(urls)


def make_image_block_from_upload(uploaded_file: Any) -> dict[str, Any]:
    data = uploaded_file.getvalue()
    media_type = infer_media_type(Path(uploaded_file.name), data)
    if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
        raise ValueError(
            f"unsupported image format: {uploaded_file.name} "
            "(jpeg/png/gif/webp only)"
        )

    import base64

    b64 = base64.b64encode(data).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64,
        },
    }


def looks_like_text_upload(uploaded_file: Any) -> bool:
    mime = str(getattr(uploaded_file, "type", "") or "").lower()
    if mime.startswith("text/"):
        return True
    if mime in {
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/yaml",
        "application/toml",
    }:
        return True
    return Path(str(uploaded_file.name or "")).suffix.lower() in TEXT_FILE_EXTENSIONS


def make_text_block_from_upload(uploaded_file: Any) -> tuple[dict[str, Any], str | None]:
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError(f"empty file: {uploaded_file.name}")
    if b"\x00" in data[:4096]:
        raise ValueError(f"binary file is not supported: {uploaded_file.name}")

    byte_truncated = False
    if len(data) > MAX_TEXT_FILE_BYTES:
        data = data[:MAX_TEXT_FILE_BYTES]
        byte_truncated = True

    decoded: str | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp932", "shift_jis", "euc_jp"):
        try:
            decoded = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError(
            "failed to decode text file (supported encodings: utf-8/cp932/shift_jis/euc_jp): "
            f"{uploaded_file.name}"
        )

    char_truncated = len(decoded) > MAX_TEXT_FILE_CHARS
    extracted = decoded[:MAX_TEXT_FILE_CHARS]
    warning: str | None = None
    if byte_truncated or char_truncated:
        warning = (
            f"text file truncated: {uploaded_file.name} "
            f"(max_bytes={MAX_TEXT_FILE_BYTES}, max_chars={MAX_TEXT_FILE_CHARS})"
        )

    block = {
        "type": "text",
        "text": f"[File reference] {uploaded_file.name}\n{extracted}",
    }
    return block, warning


def estimate_input_tokens(system_prompt: str, messages: list[dict[str, Any]]) -> int:
    token_budget = math.ceil(len(system_prompt) / TOKEN_ESTIMATE_CHARS_PER_TOKEN)
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            token_budget += math.ceil(len(content) / TOKEN_ESTIMATE_CHARS_PER_TOKEN)
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type", ""))
                if block_type == "text":
                    text = str(block.get("text", ""))
                    token_budget += math.ceil(len(text) / TOKEN_ESTIMATE_CHARS_PER_TOKEN)
                elif block_type == "image":
                    token_budget += TOKEN_ESTIMATE_IMAGE_TOKENS
    return max(1, token_budget)


def count_input_tokens(
    *,
    client: Anthropic,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> tuple[int, bool]:
    try:
        result = client.messages.count_tokens(
            model=model,
            system=system_prompt,
            messages=messages,
        )
        return int(result.input_tokens), True
    except Exception:  # noqa: BLE001
        return estimate_input_tokens(system_prompt, messages), False


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    max_chars = max(120, max_chars)
    marker = "\n...[truncated]...\n"
    marker_len = len(marker)
    if max_chars <= marker_len + 2:
        return text[:max_chars]
    remain = max_chars - marker_len
    head = remain // 2
    tail = remain - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def fit_request_to_input_budget(
    *,
    client: Anthropic,
    model: str,
    system_prompt: str,
    runtime_messages: list[dict[str, Any]],
    user_content_blocks: list[dict[str, Any]],
    max_input_tokens: int,
) -> tuple[list[dict[str, Any]], list[str], int, bool]:
    working_history = copy.deepcopy(runtime_messages)
    working_user_blocks = copy.deepcopy(user_content_blocks)

    def build_messages() -> list[dict[str, Any]]:
        return working_history + [{"role": "user", "content": working_user_blocks}]

    notes: list[str] = []
    dropped_history_turns = 0
    dropped_attachment_blocks = 0
    trimmed_any_attachment = False
    user_text_trimmed = False

    request_messages = build_messages()
    input_tokens, used_exact_count = count_input_tokens(
        client=client,
        model=model,
        system_prompt=system_prompt,
        messages=request_messages,
    )
    if input_tokens <= max_input_tokens:
        return request_messages, notes, input_tokens, used_exact_count

    for _ in range(40):
        changed = False
        attachment_indices: list[int] = []
        largest_attachment_idx = -1
        largest_attachment_len = 0

        for idx, block in enumerate(working_user_blocks):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = str(block.get("text", ""))
            if text.startswith(ATTACHMENT_TEXT_PREFIXES):
                attachment_indices.append(idx)
                if len(text) > largest_attachment_len:
                    largest_attachment_len = len(text)
                    largest_attachment_idx = idx

        if largest_attachment_idx >= 0 and largest_attachment_len > MIN_ATTACHMENT_TEXT_CHARS:
            current_text = str(working_user_blocks[largest_attachment_idx].get("text", ""))
            target_chars = max(MIN_ATTACHMENT_TEXT_CHARS, int(len(current_text) * 0.72))
            reduced_text = _truncate_middle(
                current_text,
                target_chars,
            )
            if reduced_text != current_text:
                working_user_blocks[largest_attachment_idx]["text"] = reduced_text
                trimmed_any_attachment = True
                changed = True
        elif attachment_indices:
            del working_user_blocks[attachment_indices[-1]]
            dropped_attachment_blocks += 1
            changed = True
        elif len(working_history) >= 2:
            del working_history[:2]
            dropped_history_turns += 1
            changed = True
        else:
            user_text_idx = -1
            user_text_len = 0
            for idx, block in enumerate(working_user_blocks):
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = str(block.get("text", ""))
                if text.startswith(ATTACHMENT_TEXT_PREFIXES):
                    continue
                if len(text) >= user_text_len:
                    user_text_idx = idx
                    user_text_len = len(text)
            if user_text_idx >= 0 and user_text_len > MIN_USER_TEXT_CHARS:
                current_text = str(working_user_blocks[user_text_idx].get("text", ""))
                target_chars = max(MIN_USER_TEXT_CHARS, int(len(current_text) * 0.8))
                reduced_text = _truncate_middle(
                    current_text,
                    target_chars,
                )
                if reduced_text != current_text:
                    working_user_blocks[user_text_idx]["text"] = reduced_text
                    user_text_trimmed = True
                    changed = True

        if not changed:
            break

        request_messages = build_messages()
        input_tokens, used_exact_count = count_input_tokens(
            client=client,
            model=model,
            system_prompt=system_prompt,
            messages=request_messages,
        )
        if input_tokens <= max_input_tokens:
            break

    if trimmed_any_attachment:
        notes.append("添付テキストを自動圧縮しました。")
    if dropped_attachment_blocks > 0:
        notes.append(f"添付テキストを自動除外しました: {dropped_attachment_blocks} block(s)")
    if dropped_history_turns > 0:
        notes.append(f"古い履歴を自動削減しました: {dropped_history_turns} turn(s)")
    if user_text_trimmed:
        notes.append("入力文を一部圧縮してトークン予算に合わせました。")
    if input_tokens > max_input_tokens:
        notes.append(
            f"入力トークンが予算超過のままです: {input_tokens} > {max_input_tokens}"
        )

    return request_messages, notes, input_tokens, used_exact_count


def build_user_content(
    *,
    user_text: str,
    uploaded_images: list[Any],
    uploaded_text_files: list[Any],
    manual_image_urls: list[str],
    manual_web_urls: list[str],
    allow_insecure_ssl: bool,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    auto_urls = extract_urls(user_text)
    auto_image_urls = [url for url in auto_urls if looks_like_image_url(url)]
    auto_web_urls = [url for url in auto_urls if url not in auto_image_urls]

    image_urls = dedupe_keep_order(manual_image_urls + auto_image_urls)
    web_urls = dedupe_keep_order(manual_web_urls + auto_web_urls)

    warnings: list[str] = []
    content: list[dict[str, Any]] = []
    summary_lines: list[str] = []

    # Image files first, then image URLs. Limit total count.
    image_slots = MAX_IMAGES_PER_TURN
    for uploaded in uploaded_images:
        if image_slots <= 0:
            break
        try:
            content.append(make_image_block_from_upload(uploaded))
            summary_lines.append(f"- image_file: {uploaded.name}")
            image_slots -= 1
        except Exception as e:  # noqa: BLE001
            warnings.append(str(e))

    if image_slots <= 0 and image_urls:
        warnings.append(f"image inputs truncated to {MAX_IMAGES_PER_TURN} per turn")

    for image_url in image_urls[:image_slots]:
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

    if len(uploaded_text_files) > MAX_TEXT_FILES_PER_TURN:
        warnings.append(f"text files truncated to {MAX_TEXT_FILES_PER_TURN} per turn")

    for uploaded in uploaded_text_files[:MAX_TEXT_FILES_PER_TURN]:
        if not looks_like_text_upload(uploaded):
            warnings.append(
                f"unsupported file type (text files only): {uploaded.name}"
            )
            continue
        try:
            block, warning = make_text_block_from_upload(uploaded)
            content.append(block)
            summary_lines.append(f"- text_file: {uploaded.name}")
            if warning:
                warnings.append(warning)
        except Exception as e:  # noqa: BLE001
            warnings.append(str(e))

    if len(web_urls) > MAX_WEB_URLS_PER_TURN:
        warnings.append(f"web urls truncated to {MAX_WEB_URLS_PER_TURN} per turn")
    web_urls = web_urls[:MAX_WEB_URLS_PER_TURN]

    for web_url in web_urls:
        try:
            text = fetch_webpage_text(web_url, allow_insecure_ssl=allow_insecure_ssl)
            content.append(
                {
                    "type": "text",
                    "text": f"[Web reference] {web_url}\n{text}",
                }
            )
            summary_lines.append(f"- web_url: {web_url}")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"failed to fetch url {web_url}: {e}")

    content.append({"type": "text", "text": user_text})

    user_summary = user_text
    if summary_lines:
        user_summary = f"{user_text}\n\n[Attached context]\n" + "\n".join(summary_lines)

    return content, user_summary, warnings


def generate_assistant_reply(
    *,
    client: Anthropic,
    model: str,
    max_tokens: int,
    system_prompt: str,
    request_messages: list[dict[str, Any]],
) -> tuple[str, int, int, str | None, int]:
    assistant_parts: list[str] = []
    total_in = 0
    total_out = 0
    continuation_count = 0
    last_stop_reason = None

    live_placeholder = st.empty()
    streamed_text = ""

    while True:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=request_messages,
        ) as stream:
            for text in stream.text_stream:
                streamed_text += text
                live_placeholder.markdown(streamed_text)
            final_message = stream.get_final_message()

        part_text = extract_text(final_message)
        assistant_parts.append(part_text)

        usage = final_message.usage
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        last_stop_reason = final_message.stop_reason

        if final_message.stop_reason != "max_tokens":
            break

        if continuation_count >= MAX_AUTO_CONTINUATIONS:
            st.warning("自動継続の上限に達しました。応答が途中の可能性があります。")
            break

        if not part_text.strip():
            st.warning("継続チャンクが空だったため停止しました。")
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

    final_text = "".join(assistant_parts).strip()
    live_placeholder.markdown(final_text)
    return final_text, total_in, total_out, last_stop_reason, continuation_count


def init_state() -> None:
    if st.session_state.get("ui_initialized"):
        if "theme_mode" not in st.session_state:
            st.session_state.theme_mode = "Dark"
        return

    load_dotenv(override=False)
    db_path = DEFAULT_DB_PATH
    store = MemoryStore(db_path)

    latest = store.get_latest_session_id()
    if latest:
        session_id = latest
        runtime_messages = store.load_session_messages(latest, DEFAULT_MAX_HISTORY_TURNS)
        resumed = True
    else:
        session_id = make_session_id()
        runtime_messages = []
        resumed = False

    st.session_state.ui_initialized = True
    st.session_state.db_path = db_path
    st.session_state.model = DEFAULT_MODEL
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    st.session_state.max_input_tokens = DEFAULT_MAX_INPUT_TOKENS
    st.session_state.max_history_turns = DEFAULT_MAX_HISTORY_TURNS
    st.session_state.memory_top_k = DEFAULT_MEMORY_TOP_K
    st.session_state.allow_insecure_ssl = False
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    st.session_state.theme_mode = "Dark"
    st.session_state.session_id = session_id
    st.session_state.runtime_messages = runtime_messages
    st.session_state.display_messages = runtime_to_display_messages(runtime_messages)
    st.session_state.resumed = resumed


def main() -> None:
    st.set_page_config(
        page_title="Criticism Bot UI",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    apply_styles(st.session_state.theme_mode)

    st.title("Criticism Bot UI")
    st.caption("SQLite永続化・関連メモリ参照・画像/URL入力対応")

    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except StreamlitSecretNotFoundError:
        api_key = None
    if not api_key:
        import os

        api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY が見つかりません。.env を設定してください。")
        st.stop()

    with st.sidebar:
        st.subheader("Session")
        st.selectbox("Theme", options=["Dark", "Light"], key="theme_mode")
        st.text_input("Database Path", key="db_path")
        st.caption(f"Resolved DB: {Path(st.session_state.db_path).expanduser().resolve()}")

        recent_sessions = get_recent_sessions(st.session_state.db_path)
        st.caption(f"Sessions found: {len(recent_sessions)}")
        session_options = recent_sessions if recent_sessions else [st.session_state.session_id]
        if st.session_state.session_id not in session_options:
            session_options = [st.session_state.session_id] + session_options

        selected_session = st.selectbox(
            "Recent Sessions",
            options=session_options,
            index=session_options.index(st.session_state.session_id),
        )

        c1, c2 = st.columns(2)
        if c1.button("Load Session", use_container_width=True):
            st.session_state.session_id = selected_session
            store = MemoryStore(st.session_state.db_path)
            runtime = store.load_session_messages(
                selected_session,
                st.session_state.max_history_turns,
            )
            st.session_state.runtime_messages = runtime
            st.session_state.display_messages = runtime_to_display_messages(runtime)
            st.session_state.resumed = False
            st.rerun()

        if c2.button("New Session", use_container_width=True):
            st.session_state.session_id = make_session_id()
            st.session_state.runtime_messages = []
            st.session_state.display_messages = []
            st.session_state.resumed = False
            st.rerun()

        manual_session_id = st.text_input("Manual Session ID")
        if st.button("Load Manual Session", use_container_width=True):
            if not manual_session_id.strip():
                st.warning("Session IDを入力してください。")
            else:
                sid = manual_session_id.strip()
                st.session_state.session_id = sid
                store = MemoryStore(st.session_state.db_path)
                runtime = store.load_session_messages(
                    sid,
                    st.session_state.max_history_turns,
                )
                st.session_state.runtime_messages = runtime
                st.session_state.display_messages = runtime_to_display_messages(runtime)
                st.session_state.resumed = False
                if not runtime:
                    st.warning(f"指定セッションが見つからないか、履歴0件です: {sid}")
                st.rerun()

        st.markdown(
            f"<div class='meta'>Current Session: <code>{st.session_state.session_id}</code></div>",
            unsafe_allow_html=True,
        )
        if st.session_state.resumed:
            st.info("前回セッションを自動復元しています。")

        with st.expander("Edit & Rollback", expanded=False):
            store_for_edit = MemoryStore(st.session_state.db_path)
            turn_rows = store_for_edit.load_session_turn_rows(
                st.session_state.session_id,
                limit=200,
            )
            if not turn_rows:
                st.caption("このセッションには編集可能な履歴がありません。")
            else:
                turn_map = {int(row["id"]): row for row in turn_rows}
                turn_ids = list(turn_map.keys())
                selected_turn_id = st.selectbox(
                    "編集対象ターン",
                    options=turn_ids,
                    format_func=lambda tid: (
                        f"id={turn_map[tid]['id']} {turn_map[tid]['created_at']} "
                        f"{(turn_map[tid]['user_text'] or '').replace(chr(10), ' ')[:40]}"
                    ),
                )
                selected_turn = turn_map[selected_turn_id]
                edited_text = st.text_area(
                    "編集後ユーザー入力",
                    value=selected_turn["user_text"] or "",
                    height=180,
                )
                if st.button("Rollback to here and resend", use_container_width=True):
                    edited_text = edited_text.strip()
                    if not edited_text:
                        st.warning("編集後ユーザー入力が空です。")
                    else:
                        rollback_turn_id = int(selected_turn["id"])
                        runtime = store_for_edit.load_session_messages_before_turn(
                            st.session_state.session_id,
                            rollback_turn_id,
                            st.session_state.max_history_turns,
                        )
                        st.session_state.runtime_messages = runtime
                        st.session_state.display_messages = runtime_to_display_messages(
                            runtime
                        )
                        st.session_state.pending_user_text = edited_text
                        st.session_state.pending_rollback_turn_id = rollback_turn_id
                        st.session_state.resumed = False
                        st.toast("Rollbackを準備しました。再生成します。")
                        st.rerun()

        st.divider()
        st.subheader("Model")
        st.text_input("Model", key="model")
        st.number_input("Max Tokens", min_value=256, max_value=8192, step=128, key="max_tokens")
        st.number_input(
            "Max Input Tokens",
            min_value=4096,
            max_value=MAX_INPUT_TOKENS_UI_MAX,
            step=1024,
            key="max_input_tokens",
        )
        st.number_input(
            "Max History Turns",
            min_value=1,
            max_value=200,
            step=1,
            key="max_history_turns",
        )
        st.number_input(
            "Related Memory Top-K",
            min_value=0,
            max_value=20,
            step=1,
            key="memory_top_k",
        )
        st.checkbox("Allow Insecure SSL for URL Fetch", key="allow_insecure_ssl")
        store_for_summary = MemoryStore(st.session_state.db_path)
        session_summary_row = store_for_summary.get_session_summary(
            st.session_state.session_id
        )
        summary_len = len(session_summary_row[0]) if session_summary_row else 0
        st.caption(f"Session Summary Chars: {summary_len}")

        with st.expander("System Prompt", expanded=False):
            st.text_area("", key="system_prompt", height=320, label_visibility="collapsed")

    # Chat history
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.expander("添付入力（任意）", expanded=False):
        uploaded_images = st.file_uploader(
            "画像ファイル（複数可）",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            accept_multiple_files=True,
        )
        uploaded_text_files = st.file_uploader(
            "ドキュメントファイル（ドラッグ&ドロップ、複数可）",
            accept_multiple_files=True,
            help=(
                "テキスト系ファイルをそのまま添付できます。"
                "バイナリ(PDF/Office等)は未対応です。"
            ),
        )
        image_url_text = st.text_area(
            "画像URL（改行またはスペース区切り）",
            placeholder="https://example.com/artwork.jpg",
        )
        web_url_text = st.text_area(
            "Web URL（改行またはスペース区切り）",
            placeholder="https://artist-site.example.com/work",
        )

    pending_user_text = st.session_state.pop("pending_user_text", None)
    pending_rollback_turn_id = st.session_state.get("pending_rollback_turn_id")
    is_rollback_resend = (
        pending_user_text is not None and pending_rollback_turn_id is not None
    )
    user_text = pending_user_text or st.chat_input("作品や意図を入力してください")
    if not user_text:
        return
    if pending_user_text is None:
        st.session_state.pop("pending_rollback_turn_id", None)

    model_name = str(st.session_state.model or "").strip()
    if not model_name:
        model_name = DEFAULT_MODEL
        st.warning(f"Model が空だったため `{DEFAULT_MODEL}` に自動復旧しました。")

    if is_rollback_resend:
        user_image_urls = []
        user_web_urls = []
        uploaded_images = []
        uploaded_text_files = []
        st.info("編集された過去入力から再生成しています。")
    else:
        user_image_urls = parse_url_input(image_url_text)
        user_web_urls = parse_url_input(web_url_text)
        uploaded_images = uploaded_images or []
        uploaded_text_files = uploaded_text_files or []

    with st.chat_message("user"):
        st.markdown(user_text)

    user_content_blocks, user_summary, warnings = build_user_content(
        user_text=user_text,
        uploaded_images=uploaded_images,
        uploaded_text_files=uploaded_text_files,
        manual_image_urls=user_image_urls,
        manual_web_urls=user_web_urls,
        allow_insecure_ssl=st.session_state.allow_insecure_ssl,
    )

    for warning in warnings:
        st.warning(warning)

    client = Anthropic(api_key=api_key)
    store = MemoryStore(st.session_state.db_path)
    related = store.search_related(
        user_text,
        st.session_state.memory_top_k,
        exclude_session_id=st.session_state.session_id,
    )
    session_summary_row = store.get_session_summary(st.session_state.session_id)
    session_summary = (
        session_summary_row[0].strip()
        if session_summary_row and session_summary_row[0].strip()
        else ""
    )
    memory_context = build_memory_context_block(related) if related else ""
    effective_system_prompt = st.session_state.system_prompt
    if session_summary:
        effective_system_prompt = (
            f"{effective_system_prompt}\n\n"
            "以下はこのセッションの圧縮要約です。必要な文脈として参照してください。\n"
            f"{session_summary}"
        )
    if memory_context:
        effective_system_prompt = f"{effective_system_prompt}\n\n{memory_context}"

    request_messages, budget_notes, preflight_input_tokens, token_count_exact = (
        fit_request_to_input_budget(
            client=client,
            model=model_name,
            system_prompt=effective_system_prompt,
            runtime_messages=st.session_state.runtime_messages,
            user_content_blocks=user_content_blocks,
            max_input_tokens=int(st.session_state.max_input_tokens),
        )
    )
    for note in budget_notes:
        st.warning(note)
    token_counter_label = "input_tokens" if token_count_exact else "input_tokens_est"
    st.caption(
        f"{token_counter_label}={preflight_input_tokens} / budget={int(st.session_state.max_input_tokens)}"
    )

    with st.chat_message("assistant"):
        try:
            assistant_text, total_in, total_out, stop_reason, continuations = generate_assistant_reply(
                client=client,
                model=model_name,
                max_tokens=int(st.session_state.max_tokens),
                system_prompt=effective_system_prompt,
                request_messages=request_messages,
            )
        except anthropic.APIError as e:
            if is_rollback_resend:
                st.session_state.pop("pending_rollback_turn_id", None)
            st.error(f"API error: {e}")
            return
        except Exception as e:  # noqa: BLE001
            if is_rollback_resend:
                st.session_state.pop("pending_rollback_turn_id", None)
            st.error(f"Unexpected error: {e}")
            return

        st.caption(
            f"tokens: in={total_in} out={total_out} total={total_in + total_out} "
            f"stop_reason={stop_reason} continuations={continuations}"
        )

    # Save runtime context and DB record.
    if is_rollback_resend:
        try:
            deleted = store.delete_turns_from(
                st.session_state.session_id,
                int(pending_rollback_turn_id),
            )
            st.toast(f"Rolled back {deleted} turn(s).")
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("pending_rollback_turn_id", None)
            st.error(f"rollback apply failed: {e}")
            return
        st.session_state.pop("pending_rollback_turn_id", None)

    st.session_state.runtime_messages.append({"role": "user", "content": user_summary})
    st.session_state.runtime_messages.append({"role": "assistant", "content": assistant_text})
    st.session_state.runtime_messages = trim_runtime_messages(
        st.session_state.runtime_messages,
        st.session_state.max_history_turns,
    )
    store.save_turn(st.session_state.session_id, user_summary, assistant_text)
    try:
        refresh_session_summary(
            client=client,
            memory_store=store,
            session_id=st.session_state.session_id,
            model=model_name,
            keep_recent_turns=int(st.session_state.max_history_turns),
        )
    except Exception as e:  # noqa: BLE001
        st.warning(f"summary refresh failed: {e}")

    st.session_state.display_messages.append({"role": "user", "content": user_text})
    st.session_state.display_messages.append({"role": "assistant", "content": assistant_text})

    st.rerun()


if __name__ == "__main__":
    main()
