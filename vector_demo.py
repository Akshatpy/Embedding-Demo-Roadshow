"""
SpladeX roadshow demo -- shows how the SAME query gets turned into a vector three
different ways: dense embedding, classic sparse (real NumPy-corpus IDF), and learned
sparse (the real SpladeX model, run on the query text to show its expansion).

No "search a doc" step -- purely "how does each paradigm represent this query", which
is the actual ask: point at the screen, you explain, nothing to click through.

This folder is self-contained and deployable (Streamlit Community Cloud / Hugging Face
Spaces) -- see DEPLOY.md. Locally:

    pip install -r requirements.txt
    streamlit run vector_demo.py
"""
import streamlit as st

from vector_core import (
    dense_vector,
    numpy_corpus_sparse_vector,
    splade_doc_vector,
)

st.set_page_config(page_title="Words -> Vectors", layout="wide")

PRESETS = [
    ("finite?", "test if a number is finite"),
    ("zero-fill", "replace values with zero"),
    ("average", "compute the average"),
    ("flip it", "flip an array upside down"),
    ("stack it", "stack arrays vertically"),
    ("running total", "compute cumulative sum"),
]
ACCENT = "#4F46E5"

st.markdown(
    f"""
    <style>
    #MainMenu, header, footer {{visibility:hidden;}}
    .block-container {{padding-top:2.5rem; max-width:1150px;}}
    /* force a light theme regardless of the browser/OS dark-mode setting --
       Streamlit's dark theme defaults every unstyled text element to white,
       which goes invisible on our white cards unless we pin colors explicitly. */
    body, .stApp, [data-testid="stAppViewContainer"] {{background:#FAFAFA !important; color:#1C1C1E;}}
    .stApp p, .stApp span, .stApp label, .stApp div {{color:#1C1C1E;}}
    h1 {{font-weight:600; letter-spacing:-0.5px; font-size:2.3rem; margin-bottom:1.4rem; color:#1C1C1E !important;}}
    div[data-testid="stTextInput"] input {{
        border-radius:12px; border:1.5px solid #E5E5EA; padding:13px 16px;
        font-size:1.05rem; background:white !important; color:#1C1C1E !important;
        caret-color:#1C1C1E;
    }}
    div[data-testid="stTextInput"] input::placeholder {{color:#B8B8BC !important;}}
    div[data-testid="stTextInput"] input:focus {{border-color:{ACCENT};}}
    .stButton button {{
        border-radius:10px; border:1px solid #E5E5EA; background:white !important; color:#3A3A3C !important;
        font-weight:500;
    }}
    .stButton button p {{color:#3A3A3C !important;}}
    .stButton button:hover {{border-color:{ACCENT}; color:{ACCENT} !important;}}
    .stButton button:hover p {{color:{ACCENT} !important;}}
    div[data-testid="column"] .stButton button {{
        border-radius:999px; padding:2px 14px; font-size:0.85rem;
    }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
        color:#8A8A8E !important; font-size:0.8rem !important;
    }}
    .card {{
        background:white !important; border:1px solid #EFEFF1; border-radius:16px;
        padding:22px 24px 26px 24px; height:100%;
    }}
    .label {{
        text-transform:uppercase; letter-spacing:1px; font-size:0.72rem;
        font-weight:600; color:#A0A0A5 !important; margin-bottom:2px;
    }}
    .sub {{color:#B8B8BC !important; font-size:0.78rem; margin-bottom:14px;}}
    .barcode {{display:flex; height:56px; border-radius:8px; overflow:hidden; margin:10px 0;}}
    .barseg {{flex:1;}}
    .nums {{font-family:'SFMono-Regular',Consolas,monospace; font-size:0.82rem; color:#8A8A8E !important;}}
    table.vec {{width:100%; border-collapse:collapse; margin-top:8px;}}
    table.vec td {{padding:5px 0; font-size:0.92rem; border-bottom:1px solid #F2F2F4;}}
    table.vec td.k {{color:#3A3A3C !important;}}
    table.vec td.w {{text-align:right; font-family:'SFMono-Regular',Consolas,monospace; color:#3A3A3C !important;}}
    table.vec tr.hit td.k {{color:{ACCENT} !important; font-weight:600;}}
    table.vec tr.hit td.w {{color:{ACCENT} !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="loading models...")
def warm_up():
    dense_vector("warm up")
    splade_doc_vector("warm up", top_k=5)
    numpy_corpus_sparse_vector("warm up")
    return True


warm_up()

st.title("Words -> Vectors")

if "query_text" not in st.session_state:
    st.session_state.query_text = PRESETS[0][1]
if "computed_query" not in st.session_state:
    st.session_state.computed_query = PRESETS[0][1]


def submit():
    st.session_state.computed_query = st.session_state.query_text


def pick(q):
    st.session_state.query_text = q
    st.session_state.computed_query = q


row = st.columns([5, 1])
with row[0]:
    st.text_input("query", key="query_text", label_visibility="collapsed", on_change=submit)
with row[1]:
    st.button("Embed", on_click=submit, use_container_width=True)

pills = st.columns(len(PRESETS))
for col, (label, q) in zip(pills, PRESETS):
    col.button(label, on_click=pick, args=(q,), use_container_width=True)

query = st.session_state.computed_query.strip()
if not query:
    st.stop()

st.write("")
c1, c2, c3 = st.columns(3, gap="medium")

# --------------------------------------------------------------- dense
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Dense embedding</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">MiniLM &middot; 384 numbers, every one non-zero</div>', unsafe_allow_html=True)

    dq = dense_vector(query)
    vals = dq[:64]
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    bars = "".join(
        f'<div class="barseg" style="background:{ACCENT}; opacity:{0.15 + 0.85 * (v - lo) / span:.2f}"></div>'
        for v in vals
    )
    st.markdown(f'<div class="barcode">{bars}</div>', unsafe_allow_html=True)

    sample = " ".join(f"{v:+.3f}" for v in dq[:8])
    st.markdown(f'<div class="nums">[{sample}, ...]</div>', unsafe_allow_html=True)
    st.caption("first 8 of 384 numbers \u2014 no single one means anything alone")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------- sparse (real numpy corpus)
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Classic sparse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">TF &times; IDF, real doc-frequencies from NumPy\'s own 6,639-doc corpus</div>', unsafe_allow_html=True)

    vec, vocab = numpy_corpus_sparse_vector(query)
    rows = sorted(vec.items(), key=lambda kv: -kv[1])
    trs = "".join(f'<tr><td class="k">{w}</td><td class="w">{v:.2f}</td></tr>' for w, v in rows)
    st.markdown(f'<table class="vec">{trs}</table>', unsafe_allow_html=True)
    st.caption(f"{len(vec)} of {vocab:,} NumPy-corpus terms are non-zero \u2014 everything else is exactly 0")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------- learned sparse (SpladeX)
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="label">Learned sparse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">SpladeX &middot; real model, run on this text</div>', unsafe_allow_html=True)

    top_sd, nz, total = splade_doc_vector(query, top_k=12)
    q_words = set(query.lower().split())
    rows = sorted(top_sd.items(), key=lambda kv: -kv[1])
    trs = "".join(
        f'<tr class="{"hit" if t.lstrip("#") not in q_words else ""}">'
        f'<td class="k">{t}</td><td class="w">{w:.2f}</td></tr>'
        for t, w in rows
    )
    st.markdown(f'<table class="vec">{trs}</table>', unsafe_allow_html=True)
    st.caption(f"{nz} of {total:,} vocabulary terms are non-zero \u2014 highlighted ones weren't in the query")
    st.markdown("</div>", unsafe_allow_html=True)
