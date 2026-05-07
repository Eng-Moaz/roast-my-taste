"""
The Critic — AI-powered feedback assistant
Connected to Groq via LangChain (get_roast_response)
Run with: uv run streamlit run app.py
"""

import uuid
import streamlit as st
from core.bot import get_roast_response, clear_session_history

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="The Critic",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",   # sidebar hidden by default
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:           #0f0f11;
    --surface:      #1c1c22;
    --surface-2:    #222228;
    --border:       #2e2e38;
    --accent:       #e8c97e;
    --text-primary: #f0ede6;
    --text-muted:   #7a7a8a;
    --user-bubble:  #1e1e2e;
    --ai-bubble:    #16161e;
    --radius:       14px;
    --shadow:       0 4px 24px rgba(0,0,0,0.5);
}

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
}

/* Hide everything Streamlit chrome we don't want */
#MainMenu, footer, header { visibility: hidden; }

/* Hide the sidebar collapse/expand toggle arrow button entirely */
[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
    max-width: 780px !important;
}

/* ── Header row (title + restart button side-by-side) ── */
.header-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 0.15rem;
}
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem; font-weight: 700;
    color: var(--accent); line-height: 1.1;
    margin: 0;
}
.app-subtitle {
    font-size: 0.93rem; color: var(--text-muted);
    font-weight: 300; margin-bottom: 1.8rem;
}
.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0 1.6rem; }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important; box-shadow: none !important; padding: 0 !important;
}
.bubble-user {
    background: var(--user-bubble);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) 4px var(--radius);
    padding: 0.8rem 1.1rem; font-size: 0.96rem; line-height: 1.65;
    box-shadow: var(--shadow); max-width: 86%;
    margin-left: auto; color: var(--text-primary); word-break: break-word;
}
.bubble-ai {
    background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) var(--radius) 4px;
    padding: 0.8rem 1.1rem; font-size: 0.96rem; line-height: 1.65;
    box-shadow: var(--shadow); max-width: 86%;
    color: var(--text-primary); word-break: break-word;
}
.bubble-label-user {
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-muted);
    margin-bottom: 0.3rem; text-align: right;
}
.bubble-label-ai {
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.3rem;
}

/* ── Typing indicator ── */
.typing-wrap {
    padding: 0.7rem 1rem; background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-radius: var(--radius) var(--radius) var(--radius) 4px;
    width: fit-content; display: flex; gap: 5px; align-items: center;
}
.typing-wrap span {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); display: inline-block;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing-wrap span:nth-child(2) { animation-delay: .2s; }
.typing-wrap span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce {
    0%,60%,100% { transform: translateY(0); opacity:.35; }
    30%          { transform: translateY(-6px); opacity:1; }
}

/* ══════════════════════════════════════════════
   CHAT INPUT — full dark override
   The L-bracket corner decorations are SVG
   elements drawn with currentColor. Setting
   color: var(--surface) on every wrapper level
   makes them invisible (same colour as bg).
   overflow:hidden clips any residual bleed.
   ══════════════════════════════════════════════ */

/* Bottom footer bar */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background: var(--bg) !important;
    border-top: 1px solid var(--border) !important;
}

/* Outer container */
[data-testid="stChatInputContainer"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4) !important;
    padding: 0 !important;
}
[data-testid="stChatInputContainer"] > div,
[data-testid="stChatInputContainer"] > div > div {
    background: var(--surface) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}

/* Inner wrapper */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background: var(--surface) !important;
    border: none !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* The actual textarea */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea,
[data-testid="stChatInputTextArea"] {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    caret-color: var(--accent) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.96rem !important;
    opacity: 1 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 10px !important;
    resize: none !important;
}

/* BaseWeb textarea wrapper — source of the corner brackets.
   color: var(--surface) makes the currentColor SVG decorators
   invisible (same as background). Scoped ONLY to textarea side,
   NOT the submit button container so the arrow icon stays visible. */
[data-baseweb="textarea"],
[data-baseweb="base-input"] {
    background: var(--surface) !important;
    background-color: var(--surface) !important;
    color: var(--surface) !important;   /* camouflage: bracket SVGs use currentColor */
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-image: none !important;
    border-radius: 10px !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    opacity: 1 !important;
}

/* Send button — re-applies correct text color overriding parent color:surface */
[data-testid="stChatInputSubmitButton"] button {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    color: var(--accent) !important;
    transition: opacity .2s !important;
}
[data-testid="stChatInputSubmitButton"] button:hover { opacity: 0.7 !important; }
[data-testid="stChatInputSubmitButton"] button svg,
[data-testid="stChatInputSubmitButton"] button svg path {
    fill: var(--accent) !important;
    stroke: var(--accent) !important;
}

/* ── Restart button (in header row) ── */
.stButton > button {
    background: transparent !important;
    background-color: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .82rem !important;
    padding: .3rem .85rem !important;
    transition: all .2s !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(232,201,126,.06) !important;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 3rem 1rem 1.5rem; }
.empty-state .icon { font-size: 2.6rem; display: block; margin-bottom: .7rem; }
.empty-state p { font-size: .94rem; color: var(--text-muted); line-height: 1.7; max-width: 340px; margin: 0 auto; }
.empty-state .hint { font-size: .8rem; color: var(--text-muted); margin-top: .6rem; opacity: .7; }

/* Pill quick-prompt buttons */
div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] button,
div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] > button {
    background: var(--surface-2) !important;
    background-color: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    color: var(--text-muted) !important;
    font-size: .84rem !important;
    padding: .35rem 1.2rem !important;
    width: auto !important;
    display: block !important;
    margin: 0 auto .35rem auto !important;
}
div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(232,201,126,.07) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "roasted_items" not in st.session_state:
    st.session_state.roasted_items = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None


def clear_chat():
    clear_session_history(st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.roasted_items = []
    st.session_state.session_id = str(uuid.uuid4())


def render_bubble(role: str, content: str):
    if role == "user":
        st.markdown(
            f'<div class="bubble-user">'
            f'<div class="bubble-label-user">You</div>'
            f'{content}'
            f'</div><div style="height:8px"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bubble-ai">'
            f'<div class="bubble-label-ai">🎭 The Critic</div>'
            f'{content}'
            f'</div><div style="height:8px"></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# HEADER — title + restart button in one row
# ─────────────────────────────────────────────
col_title, col_btn = st.columns([5, 1])
with col_title:
    st.markdown('<div class="app-title">The Critic</div>', unsafe_allow_html=True)
with col_btn:
    st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)  # vertical align
    if st.button("↺ Restart", key="restart_btn"):
        clear_chat()
        st.rerun()

st.markdown('<div class="app-subtitle">✦ Confess your terrible taste. He\'s waiting.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT HISTORY / EMPTY STATE
# ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
        <div class="empty-state">
            <span class="icon">🎭</span>
            <p>Tell him something you like. A movie, a song, a food.<br>
               He <em>will</em> judge you.</p>
            <p class="hint">Try one of these crimes:</p>
        </div>
    """, unsafe_allow_html=True)

    QUICK_PROMPTS = [
        ("🍕", "I love pineapple on pizza"),
        ("🎬", "My favourite movie is Transformers"),
        ("🎵", "I only listen to top 40 music"),
    ]
    for icon, text in QUICK_PROMPTS:
        if st.button(f'{icon}  "{text}"', key=f"pill_{text}"):
            st.session_state.quick_prompt = text
            st.rerun()
else:
    # Render the full saved history
    for msg in st.session_state.messages:
        render_bubble(msg["role"], msg["content"])


# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
_typed = st.chat_input("Confess something you like…")
prompt = st.session_state.pop("quick_prompt", None) or _typed

if prompt:

    # 1. Save & display the user message immediately as a live placeholder
    #    (rendered BELOW the history loop above, so no duplication)
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_bubble("user", prompt)

    # 2. Typing indicator placeholder
    typing_ph = st.empty()
    typing_ph.markdown(
        '<div class="typing-wrap"><span></span><span></span><span></span></div>',
        unsafe_allow_html=True,
    )

    # 3. Call the backend
    try:
        reply = get_roast_response(
            user_input=prompt,
            roasted_items=st.session_state.roasted_items,
            session_id=st.session_state.session_id,
        )
        st.session_state.roasted_items.append(prompt)
    except Exception as e:
        reply = f"⚠️ Something went wrong. Please try again."

    # 4. Save reply and rerun — the history loop will render everything
    #    cleanly from session_state, with no duplicates.
    typing_ph.empty()
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
