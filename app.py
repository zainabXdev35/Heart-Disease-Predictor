
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Screener",
    page_icon=":material/cardiology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"

# ---------------------------------------------------------------------------
# Constants (match the notebook)
# ---------------------------------------------------------------------------
SEX_OPTIONS = {0: "Female", 1: "Male"}  # sex is coded 0 / 1 in the dataset
CP_OPTIONS = {                           # chest pain type codes 1 to 4
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "No symptoms",
}
CP_HELP = {
    1: "Chest pain with the classic pattern of angina.",
    2: "Chest pain that only partly fits the angina pattern.",
    3: "Chest pain that is not related to the heart.",
    4: "No chest pain complaint (called asymptomatic in the dataset).",
}

AGE_MIN, AGE_MAX = 20, 90
HR_MIN, HR_MAX = 60, 220
DEFAULTS = {"in_age": 55, "in_sex": 1, "in_cp": 3, "in_hr": 150}

DEFAULT_LABELS = {0: "No disease indicated", 1: "Disease present in dataset"}

# Facts reported by the notebook. Update these if the model is retrained.
MODEL_FACTS = [
    ("Model", "Support Vector Classifier (RBF kernel) inside a scaling pipeline"),
    ("Training data", "Cleveland Heart Disease dataset, 303 patients"),
    ("Inputs", "Age, sex, chest pain type, maximum heart rate"),
    ("Test accuracy", "85.25% on 61 unseen patients"),
    ("Cross-validation accuracy", "About 72% (5 folds on the training data)"),
    ("Disease class precision / recall", "0.81 / 0.89"),
]
DATA_PATTERNS = [
    ("Women (code 0)", "25.8%"),
    ("Men (code 1)", "55.3%"),
    ("Typical angina", "30.4%"),
    ("Atypical angina", "18.0%"),
    ("Non-anginal pain", "20.9%"),
    ("No symptoms (asymptomatic)", "72.9%"),
]

# ---------------------------------------------------------------------------
# Styling: mobile first. Base rules target phones, media queries enlarge them.
# Colors are set explicitly so the page looks the same in light or dark mode.
#   ink #0F2236, deep teal #0B5D6B, bright teal #14A3B5, mist #F2F7F9,
#   positive finding #B42318, negative finding #0F766E
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --ink: #0F2236;
    --deep: #0B5D6B;
    --teal: #14A3B5;
    --mist: #F2F7F9;
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

/* Mobile first spacing */
.block-container { padding: .9rem .8rem 2.5rem !important; max-width: 1040px; }

/* Hero */
.hero {
    background:
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 600 80' preserveAspectRatio='none'><path d='M0 40 H110 L128 40 L142 12 L158 68 L172 28 L184 40 H310 L328 40 L342 12 L358 68 L372 28 L384 40 H600' fill='none' stroke='rgba(255,255,255,0.22)' stroke-width='3' stroke-linejoin='round'/></svg>") no-repeat center 88% / 100% 70px,
        linear-gradient(135deg, #0A2438 0%, var(--deep) 100%);
    border-radius: 18px;
    padding: 1.4rem 1.2rem 2.6rem;
    margin-bottom: 1rem;
}
.stApp div.hero .eyebrow { color: #8FD8E3 !important; font-size: .78rem; font-weight: 600; letter-spacing: .06em; margin: 0 0 .35rem; }
.stApp div.hero h1 {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.55rem; line-height: 1.2;
    color: #FFFFFF !important; margin: 0 0 .5rem; padding: 0;
}
.stApp div.hero p { color: #BBD6DE !important; font-size: .95rem; line-height: 1.5; margin: 0; }

/* Tabs */
[data-testid="stTabs"] button[role="tab"] { min-height: 46px; }
[data-testid="stTabs"] button[role="tab"] p { color: var(--muted) !important; font-weight: 600; font-size: .98rem; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: var(--deep) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--teal) !important; }

/* Columns become white cards */
div[data-testid="stColumn"], div[data-testid="column"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.1rem 1rem !important;
    box-sizing: border-box;
}

.card-title { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.2rem; color: var(--ink) !important; margin: 0 0 .15rem; }
.card-note { color: var(--muted) !important; font-size: .9rem; margin: 0 0 1rem; line-height: 1.45; }

/* Text colors in the main area */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 600; font-size: .98rem; }
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stMain"] [data-testid="stAlert"] p,
[data-testid="stMain"] [data-testid="stAlert"] div { color: var(--ink) !important; }
[data-testid="stMain"] [data-testid="stSpinner"] p { color: var(--ink) !important; }
[data-testid="stMain"] code { color: var(--deep) !important; background: var(--chip) !important; }

/* Sliders */
[data-testid="stSlider"] [role="slider"] { background-color: var(--teal) !important; box-shadow: 0 0 0 4px rgba(20,163,181,.2); }
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] * { color: var(--deep) !important; font-weight: 700; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: var(--muted) !important; }

/* Radio buttons become touch-friendly chips (44px tall) */
div[role="radiogroup"] { gap: .45rem; flex-wrap: wrap; }
div[role="radiogroup"] label {
    background: var(--chip); border: 1.5px solid var(--line); border-radius: 12px;
    padding: .5rem .95rem; margin: 0; min-height: 44px; display: flex; align-items: center;
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
    background: #FFFFFF !important; border: 1.5px solid var(--line) !important; color: var(--deep) !important; min-height: 46px;
}
div.stButton > button p { color: inherit !important; }
div.stButton > button[kind="primary"]:hover, div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px); box-shadow: 0 8px 18px rgba(11,93,107,.3);
}
div.stButton > button:focus-visible { outline: 3px solid var(--teal); outline-offset: 2px; }

/* Live profile */
.profile { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .9rem; }
.pchip { background: var(--chip); border: 1px solid var(--line); border-radius: 999px; padding: .35rem .8rem; font-size: .88rem; font-weight: 600; color: var(--ink); }
.pchip b { color: var(--deep); }
.hr-card { background: var(--mist); border: 1px solid var(--line); border-radius: 12px; padding: .8rem .9rem; margin-bottom: 1rem; }
.hr-row { display: flex; justify-content: space-between; font-size: .88rem; font-weight: 600; color: var(--ink); margin-bottom: .4rem; }
.hr-track { height: 10px; background: #DCE8ED; border-radius: 99px; overflow: hidden; }
.hr-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--teal), var(--deep)); transition: width .4s ease; }
.hr-note { color: var(--muted); font-size: .8rem; line-height: 1.4; margin-top: .5rem; }

/* Result */
@keyframes rise { from { opacity: 0; transform: translateY(12px) scale(.98); } to { opacity: 1; transform: none; } }
.verdict { border-radius: 14px; padding: 1.15rem 1.15rem; color: #FFFFFF; margin-bottom: .9rem; animation: rise .4s ease-out; }
.verdict.pos { background: linear-gradient(130deg, #8E1B12, var(--pos)); }
.verdict.neg { background: linear-gradient(130deg, #0B5D52, var(--neg)); }
.verdict .k { font-size: .8rem; opacity: .9; letter-spacing: .04em; }
.verdict .t { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.45rem; line-height: 1.2; margin: .2rem 0 .45rem; }
.verdict .s { font-size: .9rem; line-height: 1.45; opacity: .95; }

.recap-row { display: flex; justify-content: space-between; gap: 1rem; padding: .55rem 0; border-bottom: 1px solid #E4EDF1; font-size: .93rem; }
.recap-row:last-child { border-bottom: none; }
.recap-row .k { color: var(--muted); }
.recap-row .v { color: var(--ink); font-weight: 700; text-align: right; }

.placeholder { text-align: center; color: var(--muted); padding: 1.6rem .5rem; line-height: 1.6; font-size: .95rem; }
.placeholder b { color: var(--ink); }
.stale { background: #FFF6E5; border: 1px solid #F2D9A6; color: #6B4A00; border-radius: 12px; padding: .8rem .9rem; font-size: .9rem; line-height: 1.45; margin-bottom: .8rem; }
.disclaimer { color: var(--muted); font-size: .82rem; line-height: 1.5; margin-top: .9rem; }

/* About tab tables */
.facts { width: 100%; border-collapse: collapse; font-size: .92rem; }
.facts td { padding: .6rem .3rem; border-bottom: 1px solid #E4EDF1; color: var(--ink); vertical-align: top; }
.facts td:first-child { color: var(--muted); width: 42%; padding-right: .8rem; }
.facts td:last-child { font-weight: 600; }
.facts tr:last-child td { border-bottom: none; }

/* Expander */
[data-testid="stMain"] [data-testid="stExpander"] details { background: var(--mist); border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stMain"] [data-testid="stExpander"] summary,
[data-testid="stMain"] [data-testid="stExpander"] summary p,
[data-testid="stMain"] [data-testid="stExpander"] summary span { color: var(--ink) !important; font-weight: 600; }

/* Larger screens: roomier layout */
@media (min-width: 768px) {
    .block-container { padding: 1.6rem 1.5rem 3rem !important; }
    .hero { padding: 2.1rem 2.2rem 3rem; }
    .stApp div.hero h1 { font-size: 2.2rem; max-width: 640px; }
    .stApp div.hero p { font-size: 1.02rem; max-width: 560px; }
    div[data-testid="stColumn"], div[data-testid="column"] { padding: 1.4rem 1.5rem !important; }
    .verdict .t { font-size: 1.75rem; }
}

@media (prefers-reduced-motion: reduce) {
    .verdict, .hr-fill { animation: none; transition: none; }
}
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


def run_prediction(bundle, age, sex, cp, thalach):
    """Build a one-row table in the saved feature order and predict."""
    values = {"age": age, "sex": sex, "cp": cp, "thalach": thalach}
    features = list(bundle["features"])

    unknown = [f for f in features if f not in values]
    if unknown:
        raise ValueError("The saved model expects features this app does not collect: " + ", ".join(unknown))

    row = pd.DataFrame([[float(values[f]) for f in features]], columns=features)
    model = bundle["model"]
    prediction = int(model.predict(row)[0])

    score = None
    if hasattr(model, "decision_function"):
        try:
            score = float(model.decision_function(row)[0])
        except (ValueError, AttributeError, IndexError):
            score = None
    return prediction, score


def reset_inputs():
    """Callback for the reset button: restore the default selections."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("last", None)


# Initialise widget state once so the reset button can control it
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
<div class="eyebrow">HEART DISEASE SCREENER</div>
<h1>Check a heart disease risk profile in four quick steps</h1>
<p>Enter a few details and a trained machine learning model will tell you which side of its learned pattern the profile falls on.</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_about = st.tabs(["Predict", "About the model"])

# ---------------------------------------------------------------------------
# Predict tab
# ---------------------------------------------------------------------------
with tab_predict:
    input_col, result_col = st.columns(2, gap="medium")

    with input_col:
        st.markdown('<div class="card-title">Patient details</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-note">Adjust the controls. The profile below updates as you go.</div>', unsafe_allow_html=True)

        age = st.slider("Age (years)", AGE_MIN, AGE_MAX, key="in_age")

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
        st.caption(CP_HELP[cp])

        thalach = st.slider(
            "Maximum heart rate achieved (beats per minute)",
            HR_MIN, HR_MAX, key="in_hr",
            help="The highest heart rate reached during an exercise stress test.",
        )

        # Live profile: refreshes with every change
        st.markdown(f"""
<div class="profile">
<span class="pchip"><b>{age}</b> years</span>
<span class="pchip"><b>{SEX_OPTIONS[sex]}</b></span>
<span class="pchip"><b>{CP_OPTIONS[cp]}</b></span>
<span class="pchip"><b>{thalach}</b> bpm</span>
</div>
""", unsafe_allow_html=True)

        # Context only: this value is NOT an input of the model
        age_based_max = 220 - age
        pct = max(0, min(100, round(thalach / age_based_max * 100))) if age_based_max > 0 else 0
        st.markdown(f"""
<div class="hr-card">
<div class="hr-row"><span>Heart rate vs age-based estimate</span><span>{round(thalach / age_based_max * 100) if age_based_max > 0 else 0}%</span></div>
<div class="hr-track"><div class="hr-fill" style="width:{pct}%;"></div></div>
<div class="hr-note">General reference only (220 minus age = {age_based_max} bpm). This comparison is not used by the model.</div>
</div>
""", unsafe_allow_html=True)

        predict_clicked = st.button("Predict", type="primary", key="btn_predict")
        st.button("Reset to defaults", key="btn_reset", on_click=reset_inputs)

    # Run the model when asked
    current_key = (age, sex, cp, thalach)
    error_message = None
    bundle = None
    try:
        bundle = load_bundle()
    except Exception as exc:  # corrupt file, wrong structure, version problems
        error_message = f"The model file could not be loaded. {type(exc).__name__}: {exc}"

    if predict_clicked and bundle is not None:
        try:
            with st.spinner("Analysing the profile..."):
                prediction, score = run_prediction(bundle, age, sex, cp, thalach)
            st.session_state["last"] = {"key": current_key, "pred": prediction, "score": score}
        except Exception as exc:
            error_message = f"The prediction could not be completed. {type(exc).__name__}: {exc}"

    with result_col:
        st.markdown('<div class="card-title">Result</div>', unsafe_allow_html=True)

        if error_message:
            st.error(error_message)
        elif bundle is None:
            st.warning("The model file is missing.")
            st.write("Place `heart_disease_model.pkl` in the same folder as `app.py`, then reload the page.")
        else:
            last = st.session_state.get("last")

            if last is None:
                st.markdown(
                    '<div class="placeholder"><b>No result yet.</b><br>Set the details on this page and press <b>Predict</b>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                if last["key"] != current_key:
                    st.markdown(
                        '<div class="stale">The details have changed since the last prediction. Press <b>Predict</b> to update the result.</div>',
                        unsafe_allow_html=True,
                    )

                labels = bundle.get("target_mapping", DEFAULT_LABELS)
                positive = last["pred"] == 1
                css = "pos" if positive else "neg"
                title = labels.get(last["pred"], DEFAULT_LABELS[last["pred"]])
                sub = (
                    "For the details entered, the model places this profile on the side of patients who had heart disease in its training data."
                    if positive else
                    "For the details entered, the model places this profile on the side of patients without heart disease in its training data."
                )
                st.markdown(f"""
<div class="verdict {css}">
<div class="k">MODEL RESULT</div>
<div class="t">{title}</div>
<div class="s">{sub}</div>
</div>
""", unsafe_allow_html=True)

                k_age, k_sex, k_cp, k_hr = last["key"]
                st.markdown(f"""
<div class="recap-row"><span class="k">Age</span><span class="v">{k_age} years</span></div>
<div class="recap-row"><span class="k">Sex</span><span class="v">{SEX_OPTIONS[k_sex]}</span></div>
<div class="recap-row"><span class="k">Chest pain type</span><span class="v">{CP_OPTIONS[k_cp]}</span></div>
<div class="recap-row"><span class="k">Maximum heart rate</span><span class="v">{k_hr} bpm</span></div>
""", unsafe_allow_html=True)

                if last["score"] is not None:
                    with st.expander("Technical detail"):
                        st.write(f"Decision score: **{last['score']:.2f}**")
                        st.caption(
                            "The sign of the score decides the class: positive means disease, negative means no disease. "
                            "A larger absolute value means the profile lies further from the model's decision boundary. "
                            "It is not a probability."
                        )

            st.markdown(
                '<div class="disclaimer">Educational tool only. It is not a medical diagnosis and must not replace advice from a qualified health professional.</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    about_col, data_col = st.columns(2, gap="medium")

    with about_col:
        st.markdown('<div class="card-title">How it works</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-note">The four details you enter are scaled and passed to a Support Vector Classifier trained on historical patient records.</div>', unsafe_allow_html=True)
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in MODEL_FACTS)
        st.markdown(f'<table class="facts">{rows}</table>', unsafe_allow_html=True)

        st.markdown('<div class="card-title" style="margin-top:1.2rem;">Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
- Trained on only 303 patients from one study, so results carry real uncertainty.
- Uses just four inputs. Many clinical factors are not considered.
- The training data is about two thirds male.
- The test score (85%) and the cross-validation score (72%) differ, so true accuracy is likely somewhere between them.
""")

    with data_col:
        st.markdown('<div class="card-title">Patterns in the training data</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-note">Share of patients in the dataset who had heart disease, by group. These describe the data, not any individual.</div>', unsafe_allow_html=True)
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in DATA_PATTERNS)
        st.markdown(f'<table class="facts">{rows}</table>', unsafe_allow_html=True)

        st.markdown('<div class="card-title" style="margin-top:1.2rem;">Good to know</div>', unsafe_allow_html=True)
        st.markdown("""
- People with no chest pain complaint had the highest disease rate in this dataset. That is surprising, and it is a pattern in the data rather than a medical rule.
- The result describes which group a profile resembles. It does not estimate a personal probability.
- If you are worried about your heart health, speak to a doctor.
""")

st.caption("For educational use only. Machine-learning predictions can be wrong.")