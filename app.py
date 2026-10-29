"""Cyberbullying Detection on Social Media Text
NLP mini-project: Streamlit + NLTK + TF-IDF + Logistic Regression + rule-based phrase matching.
Run with:  python -m streamlit run app.py
"""
import glob
import os
import re

import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cyberbullying_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABELS = ["Bullying", "Not Bullying"]

st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️", layout="centered")


# ------------------------------------------------------------------ NLTK (with offline fallbacks)
def _setup_nltk():
    """Try to use NLTK resources; fall back to built-in tools if they cannot be downloaded."""
    info = {"stopwords": False, "lemmatizer": False}
    try:
        import nltk
        for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        from nltk.corpus import stopwords
        stopwords.words("english")
        info["stopwords"] = True
        from nltk.stem import WordNetLemmatizer
        WordNetLemmatizer().lemmatize("tests")
        info["lemmatizer"] = True
    except Exception:
        pass
    return info


NLTK_INFO = _setup_nltk()


def get_stopwords():
    if NLTK_INFO["stopwords"]:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    return set(ENGLISH_STOP_WORDS)


def lemmatize_word(w):
    if NLTK_INFO["lemmatizer"]:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        return lem.lemmatize(lem.lemmatize(w, "v"), "n")
    for suf, rep in [("ies", "y"), ("ing", ""), ("ed", ""), ("s", "")]:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + rep
    return w


STOP = get_stopwords()


# ------------------------------------------------------------------ preprocessing
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_nlp(text: str):
    cleaned = clean_text(text)
    tokens = cleaned.split()
    filtered = [t for t in tokens if t not in STOP]
    lemmas = [lemmatize_word(t) for t in filtered]
    return cleaned, tokens, filtered, lemmas


# ------------------------------------------------------------------ rule-based phrase matching
BULLY_PHRASES = [
    "kill yourself", "go die", "nobody likes you", "nobody wants you", "everyone hates you",
    "i will find you", "i know where you live", "watch your back", "do the world a favor",
    "make your life hell", "no one cares about you", "delete your account", "you are worthless",
    "you are pathetic", "you are ugly", "hope something bad happens",
]
SAFE_PHRASES = [
    "congratulations", "thank you", "thanks for", "happy birthday", "good luck", "great job",
    "proud of you", "well done",
]


def strong_rule_label(text: str):
    t = clean_text(text)
    for p in BULLY_PHRASES:
        if p in t:
            return "Bullying", p
    for p in SAFE_PHRASES:
        if p in t:
            return "Not Bullying", p
    return None, None


# ------------------------------------------------------------------ dataset / model
def find_dataset():
    for path in sorted(glob.glob(os.path.join(BASE, "*.csv"))):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "text" in cols and "label" in cols:
            out = df[[cols["text"], cols["label"]]].copy()
            out.columns = ["text", "label"]
            return out, os.path.basename(path)
        if "tweet_text" in cols and "cyberbullying_type" in cols:  # Kaggle dataset
            out = df[[cols["tweet_text"], cols["cyberbullying_type"]]].copy()
            out.columns = ["text", "label"]
            out["label"] = out["label"].apply(lambda x: "Not Bullying" if str(x).lower() == "not_cyberbullying" else "Bullying")
            return out, os.path.basename(path)
    return None, None


@st.cache_resource(show_spinner="Training model...")
def train_model():
    df, name = find_dataset()
    if df is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
            return joblib.load(MODEL_PATH), joblib.load(VEC_PATH), None, None, None
        return None, None, None, None, None
    df = df.dropna()
    df = df[df["label"].isin(LABELS)].copy()
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]
    X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xtr, y_tr)
    pred = model.predict(vec.transform(X_te))
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, pos_label="Bullying"),
        "Recall": recall_score(y_te, pred, pos_label="Bullying"),
        "F1-score": f1_score(y_te, pred, pos_label="Bullying"),
        "Train size": len(X_tr),
        "Test size": len(X_te),
        "Dataset": name,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vec, VEC_PATH)
    return model, vec, metrics, df["label"].value_counts(), name


def load_css():
    path = os.path.join(BASE, "app.css")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ UI
load_css()
model, vec, metrics, dist, ds_name = train_model()

SAMPLES = {
    "bully": "you are so ugly and worthless, nobody likes you. just delete your account loser",
    "safe": "congratulations on your new job! thanks for helping me with the project, really appreciate it",
}
if "text" not in st.session_state:
    st.session_state.text = ""


def set_sample(key):
    st.session_state.text = SAMPLES[key]


st.markdown(
    """
<div class="hero">
  <div class="hero-icon">🛡️</div>
  <h1>Cyberbullying Detector</h1>
  <p>NLP-Powered Bullying &amp; Not Bullying Classification</p>
  <span class="pill">Machine Learning • NLP • Text Classification</span>
</div>
<div class="info-row">
  <div class="info-card"><span>NLP TECHNIQUE</span><b>TF-IDF</b></div>
  <div class="info-card"><span>ML MODEL</span><b>Logistic Regression</b></div>
  <div class="info-card"><span>CLASSES</span><b>BULLYING / NOT BULLYING</b></div>
</div>
""",
    unsafe_allow_html=True,
)

if model is None:
    st.error("No dataset (CSV) or saved model was found. Place a dataset CSV next to app.py and restart.")
    st.stop()

st.markdown('<div class="card-title">✍️ Enter Social Media Text</div><div class="muted">Paste or type the message you want to analyze.</div>', unsafe_allow_html=True)
c0, c1, c2, _ = st.columns([1.1, 1.9, 1.6, 1.6])
c0.markdown('<div class="muted" style="padding-top:8px">Try a sample:</div>', unsafe_allow_html=True)
c1.button("⚠️ Bullying example", on_click=set_sample, args=("bully",))
c2.button("✅ Safe example", on_click=set_sample, args=("safe",))

text = st.text_area("Message", key="text", height=140, placeholder="Paste or type a social media message here...", label_visibility="collapsed")
st.markdown(f'<div class="muted small">{len(text)} characters &nbsp; {len(text.split())} words</div>', unsafe_allow_html=True)

b1, b2, _ = st.columns([1.2, 1, 4])
analyze = b1.button("🔍 Analyze Text", type="primary")
b2.button("🗑 Clear", on_click=lambda: st.session_state.update(text=""))

st.markdown('<div class="disclaimer"><b>ℹ️ Disclaimer:</b> This system performs machine-learning classification based on patterns learned from the training dataset. It cannot understand full conversational context or intent.</div>', unsafe_allow_html=True)

if analyze:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        cleaned, tokens, filtered, lemmas = preprocess_nlp(text)
        X = vec.transform([cleaned])
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        p_bully = float(proba[classes.index("Bullying")])
        p_safe = float(proba[classes.index("Not Bullying")])
        label = "Bullying" if p_bully >= p_safe else "Not Bullying"
        rule_label, rule_phrase = strong_rule_label(text)
        if rule_label:
            label = rule_label
        conf = max(p_bully, p_safe)
        level = "High" if conf >= 0.8 else "Medium" if conf >= 0.6 else "Low"
        is_b = label == "Bullying"
        st.markdown(
            f"""
<div class="result {'bad' if is_b else 'good'}">
  <div class="big">{'⚠️' if is_b else '✅'}</div>
  <div class="verdict">{'BULLYING' if is_b else 'NOT BULLYING'}</div>
  <div class="muted">The model classified this text as <b>{label.upper()}</b>.</div>
  <div class="conf">Confidence: {level} ({conf*100:.2f}%)</div>
  <div class="bar-row"><span>🚨 Bullying Probability</span><b>{p_bully*100:.2f}%</b></div>
  <div class="bar"><div class="fill red" style="width:{p_bully*100:.2f}%"></div></div>
  <div class="bar-row"><span>✅ Not Bullying Probability</span><b>{p_safe*100:.2f}%</b></div>
  <div class="bar"><div class="fill green" style="width:{p_safe*100:.2f}%"></div></div>
  <div class="muted small" style="margin-top:12px">Model: TF-IDF + Logistic Regression + phrase rules • 2 classes (BULLYING / NOT BULLYING)</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if rule_label:
            st.info(f"Rule-based phrase matched: “{rule_phrase}” → {rule_label} (rule takes priority over the model).")

        with st.expander("NLP processing results", expanded=True):
            st.write("**Original text:**", text)
            st.write("**Cleaned text:**", cleaned or "-")
            st.write("**Tokens:**", tokens)
            st.write("**After stop-word removal:**", filtered)
            st.write("**After lemmatization:**", lemmas)
        with st.expander("TF-IDF feature information"):
            st.write(f"Vocabulary size: **{len(vec.vocabulary_)}** features (unigrams + bigrams)")
            row = X.tocoo()
            names = vec.get_feature_names_out()
            feats = sorted([(names[j], round(v, 4)) for j, v in zip(row.col, row.data)], key=lambda x: -x[1])[:10]
            st.write(f"Non-zero features in this text: **{X.nnz}**")
            if feats:
                st.dataframe(pd.DataFrame(feats, columns=["Feature", "TF-IDF weight"]), hide_index=True)

if metrics:
    with st.expander("Model evaluation metrics"):
        m = st.columns(4)
        for col, k in zip(m, ["Accuracy", "Precision", "Recall", "F1-score"]):
            col.metric(k, f"{metrics[k]*100:.2f}%")
        st.caption(f"Dataset: {metrics['Dataset']} • train {metrics['Train size']} / test {metrics['Test size']} (80:20 stratified split)")
    with st.expander("Dataset distribution"):
        st.bar_chart(dist)

st.markdown(
    """
<div class="card-title" style="margin-top:18px">🔎 How It Works</div>
<div class="steps">
  <div><small>01</small><b>Text Input</b><span>Enter a message</span></div><i>→</i>
  <div><small>02</small><b>Preprocessing</b><span>Clean the text</span></div><i>→</i>
  <div><small>03</small><b>TF-IDF</b><span>Extract features</span></div><i>→</i>
  <div><small>04</small><b>Logistic Regression</b><span>Classify text</span></div>
</div>
<div class="footer">Made by <b>Kaif Reshamwala</b><br>Cyberbullying Detection Using NLP • TF-IDF + Logistic Regression</div>
""",
    unsafe_allow_html=True,
)
