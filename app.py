
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon=":material/cardiology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_model.pkl"

# ---------------------------------------------------------------------------
# Constants (match the notebook's coding of the dataset)
# ---------------------------------------------------------------------------
SEX_OPTIONS = {0: "Female", 1: "Male"}
CP_OPTIONS = {
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "No symptoms",
}
AGE_MIN, AGE_MAX = 20, 90
HR_MIN, HR_MAX = 60, 220
DEFAULTS = {"in_age": 55, "in_hr": 150, "in_sex": 1, "in_cp": 3}

# ---------------------------------------------------------------------------
# Styling
# One white card centered on a soft background, identical on phone and desktop.
# Colors are set explicitly so the look does not depend on light or dark mode.
#   ink #0F2236, deep teal #0B5D6B, bright teal #14A3B5, mist #EDF4F7
#   disease present #B42318, no disease #0F766E
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ink: #0F2236;
    --deep: #0B5D6B;
    --teal: #14A3B5;
    --mist: #EDF4F7;
    --line: #D5E3E8;
    --chip: #E9F2F5;
    --muted: #52687A;
    --pos: #B42318;
    --neg: #0F766E;
}

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--mist) !important; color: var(--ink); color-scheme: light; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }

/* The page content itself is the card */
.block-container {
    max-width: 720px !important;
    margin: .7rem auto 1.5rem !important;
    padding: .9rem .9rem 1.4rem !important;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(15,34,54,.07);
}

/* Hero */
.hero {
    background:
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 600 80' preserveAspectRatio='none'><path d='M0 40 H110 L128 40 L142 12 L158 68 L172 28 L184 40 H310 L328 40 L342 12 L358 68 L372 28 L384 40 H600' fill='none' stroke='rgba(255,255,255,0.22)' stroke-width='3' stroke-linejoin='round'/></svg>") no-repeat center 90% / 100% 60px,
        linear-gradient(135deg, #0A2438 0%, var(--deep) 100%);
    border-radius: 14px;
    padding: 1.3rem 1.1rem 2.5rem;
    margin-bottom: 1.3rem;
}
.stApp div.hero h1 {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.6rem; line-height: 1.2;
    color: #FFFFFF !important; margin: 0 0 .4rem; padding: 0;
}
.stApp div.hero p { color: #BBD6DE !important; font-size: .95rem; line-height: 1.45; margin: 0; }

/* Text colors in the main area */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 600; font-size: .98rem; }
[data-testid="stMain"] [data-testid="stAlert"] p,
[data-testid="stMain"] [data-testid="stAlert"] div { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stSpinner"] p { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: var(--muted) !important; text-align: center; }
[data-testid="stMain"] code { color: var(--deep) !important; background: var(--chip) !important; }

/* Sliders */
[data-testid="stSlider"] [role="slider"] { background-color: var(--teal) !important; box-shadow: 0 0 0 4px rgba(20,163,181,.2); }
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * { color: var(--deep) !important; font-weight: 700; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }

/* Radio buttons become touch-friendly chips (44px tall) */
div[role="radiogroup"] { gap: .45rem; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: var(--chip); border: 1.5px solid var(--line); border-radius: 12px;
    padding: .5rem 1rem; margin: 0; min-height: 44px; display: flex; align-items: center;
    cursor: pointer; transition: border-color .15s ease, background .15s ease, transform .1s ease;
}
div[role="radiogroup"] label > div:first-child { display: none; }
div[role="radiogroup"] label p, div[role="radiogroup"] label span { color: var(--ink) !important; font-weight: 600; font-size: .95rem; }
div[role="radiogroup"] label:hover { border-color: var(--teal); }
div[role="radiogroup"] label:active { transform: scale(.98); }
div[role="radiogroup"] label:has(input:checked) { background: var(--deep); border-color: var(--deep); }
div[role="radiogroup"] label:has(input:checked) p,
div[role="radiogroup"] label:has(input:checked) span { color: #FFFFFF !important; }
div[role="radiogroup"] label:has(input:focus-visible) { outline: 3px solid var(--teal); outline-offset: 2px; }

/* Buttons */
div.stButton > button {
    width: 100%; min-height: 52px; border-radius: 12px; font-weight: 700; font-size: 1.02rem;
    transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
    border: 0; background: linear-gradient(90deg, var(--deep), var(--teal)) !important; color: #FFFFFF !important;
}
div.stButton > button[kind="secondary"], div.stButton > button[data-testid="stBaseButton-secondary"] {
    background: #FFFFFF !important; border: 1.5px solid var(--line) !important; color: var(--deep) !important;
}
div.stButton > button p { color: inherit !important; }
div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px); box-shadow: 0 8px 18px rgba(11,93,107,.3);
}
div.stButton > button:focus-visible { outline: 3px solid var(--teal); outline-offset: 2px; }

/* Result */
@keyframes rise { from { opacity: 0; transform: translateY(12px) scale(.98); } to { opacity: 1; transform: none; } }
.verdict {
    display: flex; align-items: center; gap: .9rem;
    border-radius: 14px; padding: 1.2rem 1.2rem; color: #FFFFFF;
    margin-top: 1.1rem; animation: rise .4s ease-out;
}
.verdict.pos { background: linear-gradient(130deg, #8E1B12, var(--pos)); }
.verdict.neg { background: linear-gradient(130deg, #0B5D52, var(--neg)); }
.verdict .dot { width: 14px; height: 14px; border-radius: 50%; background: #FFFFFF; flex: none; box-shadow: 0 0 0 5px rgba(255,255,255,.28); }
.verdict .t { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.6rem; line-height: 1.2; }
.verdict .k { font-size: .78rem; letter-spacing: .06em; opacity: .9; margin-bottom: .1rem; }

/* Larger screens */
@media (min-width: 768px) {
    .block-container { margin: 1.6rem auto 2.5rem !important; padding: 1.4rem 1.6rem 1.8rem !important; }
    .hero { padding: 1.8rem 1.7rem 2.9rem; }
    .stApp div.hero h1 { font-size: 2rem; }
    .stApp div.hero p { font-size: 1rem; }
    .verdict .t { font-size: 1.9rem; }
}

@media (prefers-reduced-motion: reduce) { .verdict { animation: none; } }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle():
    """Load the saved model bundle once. Returns None if the file is missing."""
    if not MODEL_PATH.exists():
        return None
    # Only load pickle files you created and trust.
    with MODEL_PATH.open("rb") as file:
        bundle = pickle.load(file)

    missing = {"model", "features"}.difference(bundle.keys())
    if missing:
        raise ValueError("Model file is missing: " + ", ".join(sorted(missing)))
    return bundle


def run_prediction(bundle, age, sex, cp, thalach) -> int:
    """Build a one-row table in the saved feature order and predict 0 or 1."""
    values = {"age": age, "sex": sex, "cp": cp, "thalach": thalach}
    features = list(bundle["features"])

    unknown = [f for f in features if f not in values]
    if unknown:
        raise ValueError("The saved model expects features this app does not collect: " + ", ".join(unknown))

    row = pd.DataFrame([[float(values[f]) for f in features]], columns=features)
    return int(bundle["model"].predict(row)[0])


def reset_inputs():
    """Callback for the reset button: restore default selections, clear result."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("last", None)


for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
<h1>Heart Disease Predictor</h1>
<p>Enter four details to see the model's prediction.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2, gap="large")
with c1:
    age = st.slider("Age (years)", AGE_MIN, AGE_MAX, key="in_age")
with c2:
    thalach = st.slider("Maximum heart rate (bpm)", HR_MIN, HR_MAX, key="in_hr")

sex = st.radio(
    "Sex",
    options=list(SEX_OPTIONS.keys()),
    format_func=lambda c: SEX_OPTIONS[c],
    horizontal=True,
    key="in_sex",
)

cp = st.radio(
    "Chest pain type",
    options=list(CP_OPTIONS.keys()),
    format_func=lambda c: CP_OPTIONS[c],
    horizontal=True,
    key="in_cp",
)

st.write("")
b1, b2 = st.columns([3, 1], gap="small")
with b1:
    predict_clicked = st.button("Predict", type="primary", key="btn_predict")
with b2:
    st.button("Reset", key="btn_reset", on_click=reset_inputs)

# ---------------------------------------------------------------------------
# Prediction and result
# ---------------------------------------------------------------------------
current_key = (age, sex, cp, thalach)
error_message = None
bundle = None

try:
    bundle = load_bundle()
except Exception as exc:  # corrupt file, wrong structure, version problems
    error_message = f"The model file could not be loaded. {type(exc).__name__}: {exc}"

if predict_clicked and bundle is not None:
    try:
        with st.spinner("Predicting..."):
            prediction = run_prediction(bundle, age, sex, cp, thalach)
        st.session_state["last"] = {"key": current_key, "pred": prediction}
    except Exception as exc:
        error_message = f"The prediction could not be completed. {type(exc).__name__}: {exc}"

if error_message:
    st.error(error_message)
elif bundle is None:
    st.warning("Model file not found. Place `heart_disease_model.pkl` next to `app.py` and reload.")
else:
    last = st.session_state.get("last")
    # Show the result only while it matches the current inputs
    if last is not None and last["key"] == current_key:
        positive = last["pred"] == 1
        css = "pos" if positive else "neg"
        text = "Disease present" if positive else "No disease"
        st.markdown(f"""
<div class="verdict {css}">
<div class="dot"></div>
<div>
<div class="k">PREDICTION</div>
<div class="t">{text}</div>
</div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.caption("Educational use only. Not a medical diagnosis.")
