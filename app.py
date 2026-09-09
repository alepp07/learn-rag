"""
app.py -- a contemporary bento-style chat UI over the RAG pipeline, using Streamlit.

Layout: a dark nav rail (branding, chat list, index controls), a bento-grid
stat dashboard, and the conversation. Supports a real light/dark toggle.
No new RAG concepts here -- this is ingest/embed/retrieve/generate wrapped
in a UI. Run with:

    streamlit run app.py
"""

import base64
import html
import json
import re
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from embed_store import build_index, load_index, INDEX_DIR  # noqa: E402
from generate import search, build_prompt, client, MODEL  # noqa: E402

AVATAR_PATH = Path(__file__).resolve().parent / "assets" / "bot_avatar.png"


def load_avatar_data_uri() -> str | None:
    if AVATAR_PATH.exists():
        encoded = base64.b64encode(AVATAR_PATH.read_bytes()).decode()
        return f"data:image/png;base64,{encoded}"
    return None


AVATAR_URI = load_avatar_data_uri()


def markdown_lite_to_html(text: str) -> str:
    """Escape model output, then convert a deliberate markdown subset to real
    HTML: headings, pipe tables, numbered/bulleted lists, bold/italic/code,
    and paragraphs."""
    lines = [html.escape(line) for line in text.split("\n")]
    out = []
    i, n = 0, len(lines)
    in_ul = in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < n:
        stripped = lines[i].strip()

        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|?[\s:\-]+\|[\s:\-|]*$", lines[i + 1].strip()):
            close_lists()
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            table = ['<div class="table-wrap"><table><thead><tr>']
            table += [f"<th>{c}</th>" for c in header_cells]
            table.append("</tr></thead><tbody>")
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                row_cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                table.append("<tr>" + "".join(f"<td>{c}</td>" for c in row_cells) + "</tr>")
                i += 1
            table.append("</tbody></table></div>")
            out.append("".join(table))
            continue

        heading = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if heading:
            close_lists()
            level = min(len(heading.group(1)) + 3, 6)
            out.append(f"<h{level}>{heading.group(2)}</h{level}>")
            i += 1
            continue

        numbered = re.match(r"^\d+\.\s+(.*)", stripped)
        if numbered:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{numbered.group(1)}</li>")
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{stripped[2:]}</li>")
            i += 1
            continue

        close_lists()
        if stripped:
            out.append(f"<p>{stripped}</p>")
        i += 1

    close_lists()
    joined = "\n".join(out)
    joined = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", joined)
    joined = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", joined)
    joined = re.sub(r"`(.+?)`", r"<code>\1</code>", joined)
    return joined


def relative_time(dt: datetime) -> str:
    seconds = (datetime.now() - dt).total_seconds()
    if seconds < 60:
        return "less than a minute ago"
    if seconds < 3600:
        m = int(seconds // 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if seconds < 86400:
        h = int(seconds // 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = int(seconds // 86400)
    return f"{d} day{'s' if d != 1 else ''} ago"


def format_timestamp(dt: datetime) -> str:
    if dt.date() == datetime.now().date():
        return f"Today at {dt.strftime('%H:%M')}"
    return dt.strftime("%b %-d at %H:%M")


def assistant_avatar_html() -> str:
    if AVATAR_URI:
        return f'<img src="{AVATAR_URI}" class="avatar avatar-img" alt="Assistant">'
    return '<div class="avatar avatar-assistant">A</div>'


def new_thread() -> str:
    tid = str(uuid.uuid4())
    st.session_state.threads[tid] = {"title": "New chat", "messages": [], "created": datetime.now()}
    st.session_state.active_thread_id = tid
    return tid


def ask(question: str, k: int) -> dict:
    retrieved = search(question, k=k)
    prompt = build_prompt(question, retrieved)
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    return {"content": response.choices[0].message.content, "sources": retrieved, "ts": datetime.now()}


st.set_page_config(page_title="Ask my documents", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ---------------------------------------------------------------------------
# Theme variables -- light and dark, both kept deliberately high contrast
# ---------------------------------------------------------------------------
if st.session_state.dark_mode:
    theme_vars = """
        --bg: #121417; --panel: #1B1E23; --ink: #F2F3F5; --muted: #9AA0AC;
        --border: #2A2E35; --accent: #8AA0D6; --accent-soft: #232A38; --error: #E28A7D;
        --tile-bg: #1B1E23;
    """
else:
    theme_vars = """
        --bg: #FAFAF8; --panel: #FFFFFF; --ink: #171717; --muted: #6B6B66;
        --border: #E4E3DD; --accent: #3A4B6B; --accent-soft: #EEF0F4; --error: #A33B2E;
        --tile-bg: #FFFFFF;
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {{
        {theme_vars}
        --navy: #1C2534; --navy-border: #2E394C; --navy-text: #D7DBE3; --navy-muted: #8B93A6;
        --radius-tile: 10px; --radius-btn: 6px;
    }}

    .stApp {{ background: var(--bg); color: var(--ink); font-family: 'IBM Plex Sans', sans-serif; transition: none; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 0 !important; max-width: 100% !important; }}

    [data-testid="stSidebar"] {{ background: var(--navy); border-right: 1px solid var(--navy-border); min-width: 240px !important; }}
    [data-testid="stSidebar"] * {{ color: var(--navy-text); }}
    [data-testid="stSidebar"] h3 {{ font-size: 0.8rem; font-weight: 600; color: #fff; margin: 1.2rem 0 0.5rem 0; }}
    .brand {{ display: flex; align-items: center; gap: 0.6rem; padding: 0.4rem 0 1rem 0; }}
    .brand-mark {{ width: 28px; height: 28px; border-radius: var(--radius-btn); background: var(--accent); color: #14181f; font-family: 'Source Serif 4', serif; font-weight: 600; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; }}
    .brand-name {{ font-weight: 600; font-size: 0.95rem; color: #fff; }}

    [data-testid="stSidebar"] .chatlist-time {{ font-size: 0.72rem; color: var(--navy-muted); padding-left: 0.6rem; margin: -0.35rem 0 0.6rem 0; }}
    [data-testid="stSidebar"] hr {{ border-color: var(--navy-border); margin: 1rem 0; }}

    [data-testid="stSidebar"] .stat-row {{ display: flex; justify-content: space-between; font-size: 0.78rem; padding: 0.3rem 0; border-bottom: 1px solid var(--navy-border); }}
    [data-testid="stSidebar"] .stat-row span:last-child {{ font-family: 'IBM Plex Mono', monospace; color: var(--navy-text); }}
    [data-testid="stSidebar"] .status-line {{ font-size: 0.8rem; padding-left: 0.7rem; border-left: 2px solid var(--navy-border); margin-bottom: 0.8rem; }}
    [data-testid="stSidebar"] .status-line.ready {{ border-left-color: var(--accent); color: #fff; }}

    [data-testid="stSidebar"] .stButton > button {{ background: transparent; color: var(--navy-text); border: 1px solid var(--navy-border); border-radius: var(--radius-btn); font-size: 0.8rem; width: 100%; text-align: left; transition: none; }}
    [data-testid="stSidebar"] .stButton > button:hover {{ border-color: var(--accent); color: #fff; }}
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{ background-color: var(--accent); }}
    [data-testid="stSidebar"] [data-testid="stTickBar"] {{ display: none; }}
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label {{ font-size: 0.8rem; }}

    .st-key-conv_container {{ padding: 1.4rem 2.5rem 2rem 2.5rem; }}
    .conv-title {{ font-family: 'Source Serif 4', serif; font-weight: 600; font-size: 1.4rem; color: var(--ink); }}

    .st-key-conv_container .stButton > button {{ background: var(--ink); color: var(--bg); border: 1px solid var(--ink); border-radius: var(--radius-btn); font-size: 0.8rem; padding: 0.4rem 0.9rem; transition: none; }}
    .st-key-conv_container .stButton > button:hover {{ background: var(--accent); border-color: var(--accent); color: #fff; }}

    /* --- Bento dashboard --- */
    .bento-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin: 1.2rem 0 1.6rem 0; }}
    .bento-tile {{ background: var(--tile-bg); border: 1px solid var(--border); border-radius: var(--radius-tile); padding: 1rem 1.2rem; }}
    .bento-tile.wide {{ grid-column: span 2; }}
    .bento-label {{ font-size: 0.72rem; color: var(--muted); text-transform: none; margin-bottom: 0.4rem; }}
    .bento-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 500; color: var(--ink); }}
    .bento-sub {{ font-size: 0.75rem; color: var(--muted); margin-top: 0.3rem; }}

    .msg-row {{ display: flex; gap: 0.7rem; margin-bottom: 1.3rem; align-items: flex-start; }}
    .avatar {{ width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 0.78rem; font-weight: 600; color: #fff; }}
    .avatar-img {{ object-fit: cover; background: var(--accent-soft); }}
    .avatar-assistant {{ background: var(--accent); }}
    .avatar-user {{ background: var(--ink); color: var(--bg); }}
    .msg-meta {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 0.3rem; }}

    .bubble {{ border-radius: var(--radius-tile); padding: 0.8rem 1rem; font-size: 0.92rem; line-height: 1.55; max-width: 100%; }}
    .bubble-assistant {{ background: var(--accent-soft); color: var(--ink); }}
    .bubble-user {{ background: var(--panel); border: 1px solid var(--border); color: var(--ink); display: inline-block; }}
    .bubble p, .bubble li {{ margin: 0 0 0.5em 0; }}
    .bubble p:last-child {{ margin-bottom: 0; }}
    .bubble ul, .bubble ol {{ margin: 0 0 0.6em 1.2em; padding: 0; }}
    .bubble h4, .bubble h5, .bubble h6 {{ font-size: 1em; font-weight: 600; margin: 0.9em 0 0.4em 0; }}
    .bubble h4:first-child, .bubble h5:first-child {{ margin-top: 0; }}
    .bubble code {{ background: rgba(127,127,127,0.15); padding: 0.1em 0.35em; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.85em; }}

    .table-wrap {{ overflow-x: auto; margin: 0.6em 0; }}
    .bubble table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
    .bubble th, .bubble td {{ border: 1px solid var(--border); padding: 0.4em 0.7em; text-align: left; white-space: nowrap; }}
    .bubble th {{ background: rgba(127,127,127,0.12); font-weight: 600; }}
    .bubble tr:nth-child(even) td {{ background: rgba(127,127,127,0.04); }}

    .action-btn {{ font-size: 0.72rem; color: var(--muted); background: none; border: 1px solid var(--border); border-radius: var(--radius-btn); padding: 0.28rem 0.6rem; cursor: pointer; transition: none; line-height: 1.2; font-family: 'IBM Plex Sans', sans-serif; }}
    .action-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    [class*="st-key-actions_"] {{ margin: 0.4rem 0 0 2.4rem; }}
    [class*="st-key-actions_"] [data-testid="column"] {{ display: flex; align-items: center; }}
    [class*="st-key-actions_"] .stButton > button {{
        background: none !important; color: var(--muted) !important; border: 1px solid var(--border) !important;
        font-size: 0.72rem !important; padding: 0.28rem 0.6rem !important; border-radius: var(--radius-btn) !important;
    }}
    [class*="st-key-actions_"] .stButton > button:hover {{ border-color: var(--accent) !important; color: var(--accent) !important; }}

    .chunk-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.7rem; }}
    .chunk-tile {{ background: var(--tile-bg); border: 1px solid var(--border); border-radius: var(--radius-tile); padding: 0.7rem 0.9rem; transition: none; }}
    .chunk-tile:hover {{ border-color: var(--accent); }}
    .chunk-head {{ display: flex; justify-content: space-between; margin-bottom: 0.3rem; }}
    .chunk-source {{ font-size: 0.76rem; font-weight: 500; }}
    .chunk-score-val {{ font-size: 0.72rem; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }}
    .chunk-bar-track {{ height: 4px; background: var(--border); border-radius: 2px; margin-bottom: 0.5rem; overflow: hidden; }}
    .chunk-bar-fill {{ height: 100%; background: var(--accent); }}
    .chunk-text {{ font-size: 0.8rem; color: var(--muted); line-height: 1.45; }}

    .disclaimer {{ font-size: 0.72rem; color: var(--muted); margin-top: 0.5rem; }}
    [data-testid="stForm"] {{ border: none; padding: 0; }}
    .st-key-conv_container [data-testid="stTextInput"] input {{ border-radius: var(--radius-btn); border: 1px solid var(--border); font-size: 0.9rem; background: var(--panel); color: var(--ink); }}
    [data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: var(--radius-tile); background: var(--panel); }}
    </style>
    """,
    unsafe_allow_html=True,
)

if "threads" not in st.session_state:
    st.session_state.threads = {}
if "active_thread_id" not in st.session_state or st.session_state.active_thread_id not in st.session_state.threads:
    new_thread()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">D</div><div class="brand-name">Docs Assistant</div></div>', unsafe_allow_html=True)

    if st.button("+ New chat", key="nav_new_chat"):
        new_thread()
        st.rerun()

    st.markdown("### Chats")
    ordered = sorted(st.session_state.threads.items(), key=lambda kv: kv[1]["created"], reverse=True)
    for tid, thread in ordered:
        is_active = tid == st.session_state.active_thread_id
        bullet = "\u25CF " if is_active else ""
        if st.button(f"{bullet}{thread['title']}", key=f"thread_btn_{tid}"):
            st.session_state.active_thread_id = tid
            st.rerun()
        st.markdown(f'<div class="chatlist-time">{relative_time(thread["created"])}</div>', unsafe_allow_html=True)

    st.markdown("---")
    dark = st.checkbox("Dark mode", value=st.session_state.dark_mode, key="dark_toggle")
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.markdown("### Index")
    index_exists = (INDEX_DIR / "vectors.npy").exists()
    total_chunks, counts = 0, Counter()
    if index_exists:
        st.markdown('<p class="status-line ready">Index is built and ready.</p>', unsafe_allow_html=True)
        _, chunk_meta = load_index()
        counts = Counter(c["source"] for c in chunk_meta)
        total_chunks = len(chunk_meta)
        st.markdown(f'<div class="stat-row"><span>Total chunks</span><span>{total_chunks}</span></div>', unsafe_allow_html=True)
        for source, count in counts.most_common():
            st.markdown(f'<div class="stat-row"><span>{source}</span><span>{count}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-line">No index yet. Add files to data and rebuild.</p>', unsafe_allow_html=True)

    if st.button("Rebuild index", key="nav_rebuild"):
        with st.spinner("Chunking, embedding, and indexing..."):
            build_index(data_dir=str(Path(__file__).resolve().parent / "data"))
        st.rerun()

    k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=4)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
with st.container(key="conv_container"):
    active = st.session_state.threads[st.session_state.active_thread_id]

    header_l, header_r1, header_r2 = st.columns([6, 1, 1])
    with header_l:
        st.markdown(f'<div class="conv-title">{html.escape(active["title"])}</div>', unsafe_allow_html=True)
    with header_r1:
        if st.button("New chat", key="conv_new_chat"):
            new_thread()
            st.rerun()
    with header_r2:
        if st.button("Delete", key="conv_delete"):
            del st.session_state.threads[st.session_state.active_thread_id]
            if not st.session_state.threads:
                new_thread()
            else:
                st.session_state.active_thread_id = next(iter(st.session_state.threads))
            st.rerun()

    # Bento dashboard
    top_source = counts.most_common(1)[0][0] if counts else "none"
    st.markdown(
        f"""
        <div class="bento-grid">
            <div class="bento-tile wide">
                <div class="bento-label">Index status</div>
                <div class="bento-value">{"Ready" if index_exists else "Not built"}</div>
                <div class="bento-sub">{total_chunks} chunks across {len(counts)} file{'s' if len(counts) != 1 else ''}</div>
            </div>
            <div class="bento-tile">
                <div class="bento-label">Messages</div>
                <div class="bento-value">{len(active["messages"])}</div>
                <div class="bento-sub">in this chat</div>
            </div>
            <div class="bento-tile">
                <div class="bento-label">Retrieval (k)</div>
                <div class="bento-value">{k}</div>
                <div class="bento-sub">chunks per answer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not active["messages"]:
        st.markdown(
            f"""
            <div class="msg-row">
                {assistant_avatar_html()}
                <div>
                    <div class="msg-meta">Assistant &middot; now</div>
                    <div class="bubble bubble-assistant">
                        <p>\U0001F44B Hi, ask me anything about the files in <code>data/</code>.
                        I'll answer using only what's in your indexed documents and show you
                        which chunks I used.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for i, msg in enumerate(active["messages"]):
        ts = format_timestamp(msg["ts"])
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="msg-row">
                    <div class="avatar avatar-user">Y</div>
                    <div>
                        <div class="msg-meta">You &middot; {ts}</div>
                        <div class="bubble bubble-user">{markdown_lite_to_html(msg["content"])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue

        st.markdown(
            f"""
            <div class="msg-row">
                {assistant_avatar_html()}
                <div style="flex:1; min-width:0;">
                    <div class="msg-meta">Assistant &middot; {ts}</div>
                    <div class="bubble bubble-assistant">{markdown_lite_to_html(msg["content"])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key=f"actions_{st.session_state.active_thread_id}_{i}"):
            copy_payload = json.dumps(msg["content"])
            act_col1, act_col2, _ = st.columns([1, 1.4, 8])
            with act_col1:
                st.markdown(
                    f"""
                    <button class="action-btn" onclick='
                        navigator.clipboard.writeText({copy_payload});
                        this.innerText="Copied";
                        setTimeout(() => this.innerText="Copy", 1200);
                    '>Copy</button>
                    """,
                    unsafe_allow_html=True,
                )
            with act_col2:
                if st.button("Regenerate", key=f"regen_{st.session_state.active_thread_id}_{i}"):
                    question = active["messages"][i - 1]["content"]
                    with st.spinner("Regenerating..."):
                        result = ask(question, k)
                    active["messages"][i] = {"role": "assistant", **result}
                    st.rerun()

        if msg.get("sources"):
            with st.expander(f"Retrieved {len(msg['sources'])} chunks"):
                tiles = []
                for r in msg["sources"]:
                    bar_width = max(0, min(100, round(r["score"] * 100)))
                    snippet = r["text"][:280] + ("..." if len(r["text"]) > 280 else "")
                    tiles.append(
                        f"""
                        <div class="chunk-tile">
                            <div class="chunk-head">
                                <span class="chunk-source">{html.escape(r['source'])}</span>
                                <span class="chunk-score-val">{r['score']:.2f}</span>
                            </div>
                            <div class="chunk-bar-track"><div class="chunk-bar-fill" style="width:{bar_width}%;"></div></div>
                            <div class="chunk-text">{html.escape(snippet)}</div>
                        </div>
                        """
                    )
                st.markdown(f'<div class="chunk-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)

    with st.form(key=f"ask_form_{st.session_state.active_thread_id}", clear_on_submit=True, border=False):
        in_col, btn_col = st.columns([6, 1])
        with in_col:
            question = st.text_input("Ask", placeholder="Ask anything about your documents...", label_visibility="collapsed")
        with btn_col:
            submitted = st.form_submit_button("Send")

    st.markdown(
        '<p class="disclaimer">Answers are generated from your indexed documents and may be incomplete. '
        "Retrieved sources are shown below each answer.</p>",
        unsafe_allow_html=True,
    )

    if submitted and question.strip():
        if not index_exists:
            st.markdown('<p class="status-line error">Build the index first, using the sidebar.</p>', unsafe_allow_html=True)
            st.stop()

        active["messages"].append({"role": "user", "content": question, "ts": datetime.now()})
        if active["title"] == "New chat":
            active["title"] = question[:40] + ("..." if len(question) > 40 else "")

        with st.spinner("Retrieving and generating..."):
            result = ask(question, k)
        active["messages"].append({"role": "assistant", **(result if show_chunks else {"content": result["content"], "ts": result["ts"]})})
        st.rerun()