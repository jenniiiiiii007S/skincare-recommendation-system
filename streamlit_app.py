import os
import sys
import tempfile
import html as html_lib
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google import genai
from data_loader import load_and_clean_data, build_vector_database
from pipeline import full_pipeline

st.set_page_config(
    page_title="SkinAgent — Personalized Skincare AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

:root {
  --cream:         #faf8f4;
  --cream-dark:    #f2ede4;
  --sage:          #4a7c59;
  --sage-light:    #eaf2ec;
  --sand:          #b89a72;
  --sand-light:    #f7f0e6;
  --text:          #252320;
  --text-mid:      #5a574f;
  --text-light:    #8f8b83;
  --morning:       #c47a20;
  --morning-bg:    #fdf4e7;
  --evening:       #5b4d8a;
  --evening-bg:    #f1eef8;
  --border:        #e8e2d8;
  --white:         #ffffff;
}

/* ── Base ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--cream) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Tabs ─────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 2px solid var(--border) !important;
  gap: 0.25rem;
}
[data-testid="stTabs"] button[role="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  color: var(--text-light) !important;
  padding: 0.5rem 1.1rem !important;
  border-radius: 6px 6px 0 0 !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--sage) !important;
  border-bottom: 2px solid var(--sage) !important;
  background: var(--sage-light) !important;
}

/* ── Chat messages ────────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
  background: var(--white) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important;
  color: var(--text) !important;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  background: transparent !important;
  border: 1.5px solid var(--border) !important;
  color: var(--text-mid) !important;
  border-radius: 8px !important;
  padding: 0.3rem 0.8rem !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: var(--sage) !important;
  color: var(--sage) !important;
}

/* ── File uploader ────────────────────────── */
[data-testid="stFileUploader"] > div {
  background: var(--sand-light) !important;
  border: 1.5px dashed var(--sand) !important;
  border-radius: 10px !important;
  padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  color: var(--text-mid) !important;
}

/* ── Expander ─────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--sand-light) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  color: var(--text-mid) !important;
}

/* ── Routine cards ────────────────────────── */
.routine-section {
  margin: 1.4rem 0 0.8rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 1.5px solid var(--border);
}
.section-name {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.45rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.section-name.morning { color: var(--morning); }
.section-name.evening { color: var(--evening); }

.r-card {
  background: var(--white);
  border: 1.5px solid var(--border);
  border-left: 4px solid var(--border);
  border-radius: 10px;
  padding: 0.85rem 1rem;
  margin: 0.6rem 0;
}
.r-card.morning { border-left-color: var(--morning); }
.r-card.evening { border-left-color: var(--evening); }

.r-type {
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-light);
  margin-bottom: 0.2rem;
}
.r-name {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text);
  line-height: 1.3;
}
.r-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 0.25rem;
}
.r-brand { font-size: 0.78rem; color: var(--text-light); }
.r-price {
  font-size: 0.75rem;
  font-weight: 600;
  background: var(--sand-light);
  color: var(--sand);
  padding: 1px 8px;
  border-radius: 20px;
}
.r-why {
  font-size: 0.82rem;
  color: var(--text-mid);
  line-height: 1.6;
  margin-top: 0.55rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
}

/* ── Profile chips ────────────────────────── */
.profile-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0.5rem 0 0.8rem 0;
}
.chip {
  font-size: 0.76rem;
  font-weight: 500;
  padding: 3px 11px;
  border-radius: 20px;
  background: var(--sage-light);
  color: var(--sage);
  border: 1px solid #c4dbc9;
}
.chip.allergy {
  background: var(--morning-bg);
  color: var(--morning);
  border-color: #e8c48a;
}
.chip.budget {
  background: var(--sand-light);
  color: var(--sand);
  border-color: #ddc9aa;
}

/* ── Image observation ────────────────────── */
.img-obs {
  background: var(--sand-light);
  border-left: 3px solid var(--sand);
  border-radius: 0 8px 8px 0;
  padding: 0.6rem 0.9rem;
  font-size: 0.84rem;
  color: var(--text);
  margin-bottom: 0.8rem;
  line-height: 1.55;
}
.img-obs strong { color: var(--sand); }

/* ── Warnings / Notes ─────────────────────── */
.block-card {
  border-radius: 10px;
  padding: 0.75rem 1rem;
  margin: 0.7rem 0;
}
.block-card.warn {
  background: #fff8f0;
  border: 1px solid #f0c080;
}
.block-card.note {
  background: var(--sage-light);
  border: 1px solid #c4dbc9;
}
.block-title {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.4rem;
}
.block-card.warn .block-title { color: var(--morning); }
.block-card.note .block-title { color: var(--sage); }
.block-card ul {
  margin: 0;
  padding-left: 1.1rem;
}
.block-card li {
  font-size: 0.83rem;
  color: var(--text);
  line-height: 1.7;
}

/* ── About page ───────────────────────────── */
.hero-wrap { padding: 1.5rem 0 1rem 0; }
.hero-label {
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.15em;
  color: var(--sage);
  margin-bottom: 0.4rem;
}
.hero-title {
  font-family: 'Cormorant Garamond', serif;
  font-size: 2.6rem;
  font-weight: 600;
  line-height: 1.15;
  color: var(--text);
  margin-bottom: 0.6rem;
}
.hero-sub {
  font-size: 0.95rem;
  color: var(--text-mid);
  line-height: 1.65;
  max-width: 560px;
}
.hero-team {
  font-size: 0.78rem;
  color: var(--text-light);
  margin-top: 0.75rem;
}
.divider { height: 1px; background: var(--border); margin: 1.5rem 0; }
.about-h {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text);
  margin: 1.8rem 0 0.8rem 0;
}
.pipe-step {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 0.65rem 0;
  border-bottom: 1px solid var(--border);
}
.pipe-step:last-child { border-bottom: none; }
.pipe-num {
  min-width: 26px;
  height: 26px;
  background: var(--sage);
  color: white;
  border-radius: 50%;
  font-size: 0.75rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 2px;
}
.pipe-name { font-weight: 600; font-size: 0.88rem; color: var(--text); }
.pipe-desc { font-size: 0.8rem; color: var(--text-light); line-height: 1.5; margin-top: 2px; }
.metrics-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.75rem 0; }
.m-card {
  flex: 1; min-width: 100px;
  background: var(--white);
  border: 1.5px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 0.75rem;
  text-align: center;
}
.m-val {
  font-family: 'Cormorant Garamond', serif;
  font-size: 1.9rem;
  font-weight: 600;
  color: var(--sage);
  line-height: 1;
}
.m-lbl { font-size: 0.72rem; color: var(--text-light); margin-top: 4px; line-height: 1.4; }
.feat-list { list-style: none; padding: 0; margin: 0.3rem 0; }
.feat-list li {
  font-size: 0.87rem;
  color: var(--text);
  padding: 0.32rem 0;
  display: flex;
  gap: 8px;
  align-items: flex-start;
  border-bottom: 1px solid var(--border);
}
.feat-list li:last-child { border-bottom: none; }
.feat-list li::before {
  content: "✦";
  color: var(--sage);
  font-size: 0.55rem;
  margin-top: 6px;
  flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ── Data / pipeline init ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SkinAgent…")
def init_pipeline():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY is not set.")
        st.stop()
    client = genai.Client(api_key=api_key)
    products_df, inci_df = load_and_clean_data()
    product_collection, ingredient_collection = build_vector_database(products_df, inci_df)
    return client, product_collection, ingredient_collection


client, product_collection, ingredient_collection = init_pipeline()

INITIAL_MESSAGE = {
    "role": "assistant",
    "content": (
        "Hi! Tell me about your skin — type, concerns, allergies, and budget. "
        "You can also add a facial photo below for visual analysis.\n\n"
        "*Examples:*\n"
        "- I have oily acne-prone skin and want a routine under $80\n"
        "- Dry sensitive skin, allergic to fragrance, drugstore budget\n"
        "- Anti-aging routine for combination skin, around $150"
    ),
}


# ── Routine HTML formatter ────────────────────────────────────────────────────
def _e(text):
    """HTML-escape text so Gemini backticks / asterisks render as plain text."""
    return html_lib.escape(str(text))


def format_routine_html(result):
    if "error" in result:
        return f"<p style='color:#c0392b'>{_e(result['error'])}</p>"

    parts = []
    profile = result.get("profile", {}) or {}
    budget  = result.get("budget_profile", {}) or {}
    routine = result.get("routine", {}) or {}

    # Image observation
    if profile.get("image_observations"):
        parts.append(
            f'<div class="img-obs"><strong>📸 From your photo —</strong> {_e(profile["image_observations"])}</div>'
        )

    # Profile chips
    chips = []
    if profile.get("skin_type"):
        chips.append(f'<span class="chip">{_e(profile["skin_type"])} skin</span>')
    for c in (profile.get("concerns") or []):
        chips.append(f'<span class="chip">{_e(c)}</span>')
    for a in (profile.get("allergies") or []):
        chips.append(f'<span class="chip allergy">⚠ {_e(a)}</span>')
    if budget.get("overall_limit"):
        chips.append(f'<span class="chip budget">💰 ${int(budget["overall_limit"])} budget</span>')
    elif budget.get("tier") and budget["tier"] != "any":
        chips.append(f'<span class="chip budget">💰 {_e(budget["tier"].replace("_", " "))}</span>')
    if chips:
        parts.append('<div class="profile-strip">' + "".join(chips) + '</div>')

    # Price lookup
    price_lookup = {}
    for p in result.get("retrieved_products", []):
        raw = str(p.get("price", ""))
        clean = raw.rstrip("0").rstrip(".") if "." in raw else raw
        price_lookup[p["name"]] = clean

    def render_steps(steps, time_class):
        html = ""
        for step in steps:
            name   = _e(step.get("product_name", ""))
            brand  = _e(step.get("brand", ""))
            ptype  = _e(step.get("product_type", ""))
            why    = _e(step.get("why", ""))
            price  = price_lookup.get(step.get("product_name", ""), "")
            price_html = f'<span class="r-price">{_e(price)}</span>' if price else ""
            html += f"""
            <div class="r-card {time_class}">
              <div class="r-type">{ptype}</div>
              <div class="r-name">{name}</div>
              <div class="r-meta">
                <span class="r-brand">{brand}</span>
                {price_html}
              </div>
              {"<div class='r-why'>" + why + "</div>" if why else ""}
            </div>"""
        return html

    if routine.get("morning_routine"):
        parts.append('<div class="routine-section"><span class="section-name morning">☀ Morning Routine</span></div>')
        parts.append(render_steps(routine["morning_routine"], "morning"))

    if routine.get("evening_routine"):
        parts.append('<div class="routine-section"><span class="section-name evening">🌙 Evening Routine</span></div>')
        parts.append(render_steps(routine["evening_routine"], "evening"))

    if routine.get("warnings"):
        items = "".join(f"<li>{_e(w)}</li>" for w in routine["warnings"])
        parts.append(
            f'<div class="block-card warn"><div class="block-title">⚠ Warnings</div><ul>{items}</ul></div>'
        )

    if routine.get("notes"):
        items = "".join(f"<li>{_e(n)}</li>" for n in routine["notes"])
        parts.append(
            f'<div class="block-card note"><div class="block-title">💡 Notes</div><ul>{items}</ul></div>'
        )

    if budget.get("budget_fallbacks"):
        items = "".join(
            f"<li><strong>{_e(fb['product_type'])}</strong>: cheapest available is ${_e(str(fb['actual_price']))} "
            f"(your limit was ${int(fb['user_limit'])})</li>"
            for fb in budget["budget_fallbacks"]
        )
        parts.append(
            f'<div class="block-card warn"><div class="block-title">💰 Budget Notice</div><ul>{items}</ul></div>'
        )

    return "\n".join(parts) if parts else "<p>Could not generate a routine. Try rephrasing.</p>"


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_about, tab_app = st.tabs(["🌿  About", "✨  Try It"])


# ── About tab ─────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-label">NYU DS-UA 301 · Group 12</div>
      <div class="hero-title">Personalized Skincare,<br>Powered by AI.</div>
      <div class="hero-sub">
        SkinAgent is a multimodal, retrieval-augmented multi-agent system that builds
        personalized skincare routines grounded in real products — not hallucinations.
        It understands your skin type, concerns, allergies, and budget, and checks
        ingredient safety before making any recommendation.
      </div>
      <div class="hero-team">Trinity Chan · Jennifer Ran · Savanna Thomas</div>
    </div>
    <div class="divider"></div>

    <div class="about-h">How It Works</div>
    <div>
      <div class="pipe-step">
        <div class="pipe-num">1</div>
        <div>
          <div class="pipe-name">Skin Profile Agent</div>
          <div class="pipe-desc">Parses your text description and optional facial photo into a structured skin profile using Gemini 2.5 Flash vision.</div>
        </div>
      </div>
      <div class="pipe-step">
        <div class="pipe-num">2</div>
        <div>
          <div class="pipe-name">Product Retrieval Agent</div>
          <div class="pipe-desc">Searches a ChromaDB vector database of 2,418 real products (Sephora + LookFantastic) using embedding similarity, filtered by your skin type.</div>
        </div>
      </div>
      <div class="pipe-step">
        <div class="pipe-num">3</div>
        <div>
          <div class="pipe-name">Budget Agent</div>
          <div class="pipe-desc">Extracts your price constraints and filters retrieved products by overall limit, per-category limits, or budget tier.</div>
        </div>
      </div>
      <div class="pipe-step">
        <div class="pipe-num">4</div>
        <div>
          <div class="pipe-name">Conflict Checker Agent</div>
          <div class="pipe-desc">Detects harmful ingredient combinations via hardcoded clinical rules and RAG-based Gemini reasoning. Flags allergens using synonym mapping (e.g. "fragrance" → parfum, limonene, linalool…).</div>
        </div>
      </div>
      <div class="pipe-step">
        <div class="pipe-num">5</div>
        <div>
          <div class="pipe-name">Routine Builder Agent</div>
          <div class="pipe-desc">Assembles a structured AM/PM routine using Gemini 2.5 Flash, separating conflicting ingredients across morning and evening and respecting your budget.</div>
        </div>
      </div>
    </div>

    <div class="divider"></div>
    <div class="about-h">Evaluation Results</div>
    <div class="metrics-row">
      <div class="m-card"><div class="m-val">99%</div><div class="m-lbl">Allergy detection accuracy</div></div>
      <div class="m-card"><div class="m-val">67%</div><div class="m-lbl">Conflict detection accuracy (100% recall on true conflicts)</div></div>
      <div class="m-card"><div class="m-val">83%</div><div class="m-lbl">Budget compliance accuracy</div></div>
      <div class="m-card"><div class="m-val">2,418</div><div class="m-lbl">Real products indexed</div></div>
    </div>

    <div class="divider"></div>
    <div class="about-h">Key Capabilities</div>
    <ul class="feat-list">
      <li>Accepts both text descriptions and facial photos as input</li>
      <li>Retrieves real products from Sephora and LookFantastic databases — never hallucinated</li>
      <li>Detects 5 clinically documented ingredient conflicts with 100% recall</li>
      <li>Allergen synonym mapping covers parabens, sulfates, fragrance, alcohol, silicones, and more</li>
      <li>Budget-aware filtering with per-category price limits and fallback handling</li>
      <li>Fitzpatrick fairness evaluation: 100% allergy accuracy consistency across skin tones</li>
      <li>Baseline comparison: pipeline 3.83/5 vs. raw Gemini 3.50/5 across 6 test cases</li>
    </ul>
    <div style="height: 2rem;"></div>
    """, unsafe_allow_html=True)


# ── Try It tab ────────────────────────────────────────────────────────────────
with tab_app:
    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = [INITIAL_MESSAGE]

    # Header row
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.markdown(
            '<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.3rem;'
            'font-weight:600;color:#4a7c59;margin:0.5rem 0 0.25rem 0;">SkinAgent</p>',
            unsafe_allow_html=True,
        )
    with col_reset:
        if st.button("↺ Reset"):
            st.session_state.messages = [INITIAL_MESSAGE]
            st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image_path"):
                try:
                    st.image(msg["image_path"], width=200)
                except Exception:
                    pass
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Inline image uploader
    with st.expander("📷  Add a photo of your skin (optional)"):
        uploaded = st.file_uploader(
            "Upload a clear photo of your face",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, width=180)
        st.caption("Photos are processed by Google Gemini and not stored by this app.")

    # Chat input
    user_message = st.chat_input("Describe your skin…")

    if user_message:
        image_path = save_uploaded_file(uploaded) if uploaded is not None else None

        st.session_state.messages.append({
            "role": "user",
            "content": user_message,
            "image_path": image_path,
        })
        with st.chat_message("user"):
            if image_path:
                try:
                    st.image(image_path, width=200)
                except Exception:
                    pass
            st.markdown(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Building your routine…"):
                try:
                    result = full_pipeline(
                        client, product_collection, ingredient_collection,
                        user_input=user_message,
                        image_path=image_path,
                    )
                    response_text = format_routine_html(result)
                except Exception as e:
                    response_text = f"<p style='color:#c0392b'>Something went wrong: {_e(str(e))}</p>"

            st.markdown(response_text, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
