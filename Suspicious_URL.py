import os
import socket
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

CSV_DB_PATH = "malicious_urls.csv"
CSV_COLUMNS = ["url", "label"]

#CSV DATABASE HANDLING
def normalize_url(url: str) -> str:
    """Ensure URL always has a scheme before storing or looking up."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def load_csv_db() -> pd.DataFrame:
    """
    Load the existing CSV database automatically.
    Handles both headered and headerless CSVs with url, label columns.
    Creates an empty file if the path does not exist.
    """
    if not os.path.exists(CSV_DB_PATH):
        st.warning(f"CSV not found at `{CSV_DB_PATH}`. Starting with an empty database.")
        empty = pd.DataFrame(columns=CSV_COLUMNS)
        empty.to_csv(CSV_DB_PATH, index=False)
        return empty

    # Peek at the first row to detect whether a header exists
    peek = pd.read_csv(CSV_DB_PATH, header=None, nrows=1)
    first_val = str(peek.iloc[0, 0]).strip().lower()
    has_header = first_val == "url"

    if has_header:
        df = pd.read_csv(CSV_DB_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(CSV_DB_PATH, header=None, names=["url", "label"])

    if "url" not in df.columns or "label" not in df.columns:
        st.error(
            f"CSV at `{CSV_DB_PATH}` must have `url` and `label` columns. "
            f"Found: {list(df.columns)}"
        )
        return pd.DataFrame(columns=CSV_COLUMNS)

    df["url"] = df["url"].astype(str).apply(normalize_url)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    return df[CSV_COLUMNS]

def is_in_csv_db(url: str, label: int = 1) -> bool:
    """Check if a URL exists in the CSV with the given label."""
    url = normalize_url(url)
    df = load_csv_db()
    match = df[df["url"].str.lower() == url.lower()]
    if match.empty:
        return False
    return int(match.iloc[0]["label"]) == label


#FEATURE EXTRACTION
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "bank", "secure", "update",
    "account", "confirm", "password", "billing", "suspend",
]


def contains_ip(url: str) -> int:
    try:
        host = urlparse(url).hostname or ""
        socket.inet_aton(host)
        return 1
    except OSError:
        return 0


def extract_features(url: str) -> dict:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    full = url.lower()
    parts = [p for p in hostname.split(".") if p]
    num_subdomains = max(0, len(parts) - 2)
    kw_count = sum(kw in full for kw in SUSPICIOUS_KEYWORDS)
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "has_at": int("@" in url),
        "has_dash": int("-" in hostname),
        "is_https": int(parsed.scheme == "https"),
        "suspicious_kws": kw_count,
        "has_ip": contains_ip(url),
        "num_subdomains": num_subdomains,
    }


FEATURE_COLS = [
    "url_length", "num_dots", "has_at", "has_dash",
    "is_https", "suspicious_kws", "has_ip", "num_subdomains",
]


def features_to_array(features: dict) -> np.ndarray:
    return np.array([[features[c] for c in FEATURE_COLS]])


#MODEL TRAINING
def train_model():
    """
    Train a RandomForest on the CSV data.
    Returns (model, accuracy) or (None, None) if there is not enough data.
    """
    df = load_csv_db().dropna(subset=["url", "label"])
    if df.empty:
        return None, None

    feature_rows = [extract_features(u) for u in df["url"]]
    X = pd.DataFrame(feature_rows)
    y = df["label"]

    if y.nunique() < 2 or len(df) < 6:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        return clf, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc


#PREDICTION & EXPLANATION
def predict_url(model, url: str) -> tuple[str, float, dict]:
    features = extract_features(url)
    X = features_to_array(features)
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    risk = float(proba[classes.index(1)]) if 1 in classes else 0.0

    if is_in_csv_db(url, label=1):
        risk = 1.0
    elif is_in_csv_db(url, label=0):
        risk = 0.0

    label = "Suspicious" if risk >= 0.5 else "Safe"
    return label, risk, features


def explain(features: dict, label: str, in_db: bool = False) -> list[str]:
    if label == "Safe":
        reasons = ["No major red flags detected."]
        if in_db:
            reasons.insert(0, "✅ This URL is marked safe in database.")
        return reasons

    reasons = []
    if in_db:
        reasons.append("🗂️ This URL is a known entry in database.")
    if not features["is_https"]:
        reasons.append("🔓 Not using HTTPS — data may be transmitted insecurely.")
    if features["has_ip"]:
        reasons.append("🌐 Uses a raw IP address instead of a domain name.")
    if features["has_at"]:
        reasons.append("⚠️ '@' symbol in URL can disguise the real destination.")
    if features["has_dash"]:
        reasons.append("🔤 Hyphens in the domain can indicate a spoofed brand name.")
    if features["suspicious_kws"] >= 2:
        reasons.append(f"🚨 Contains {features['suspicious_kws']} suspicious keywords.")
    elif features["suspicious_kws"] == 1:
        reasons.append("⚠️ Contains a suspicious keyword.")
    if features["url_length"] > 75:
        reasons.append(f"📏 URL is unusually long ({features['url_length']} chars).")
    if features["num_subdomains"] >= 3:
        reasons.append(f"🌲 Has {features['num_subdomains']} subdomains.")
    if not reasons:
        reasons.append("⚠️ URL resembles known phishing patterns.")
    return reasons


#STREAMLIT UI
def main():
    st.set_page_config(page_title="Suspicious URL Detector", page_icon="🔍", layout="centered")

    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "model" not in st.session_state:
        st.session_state.model = None
    if "model_acc" not in st.session_state:
        st.session_state.model_acc = None

    # Auto-load & train from the CSV on first run
    if st.session_state.model is None:
        with st.spinner(f"Loading `{CSV_DB_PATH}` and training model…"):
            st.session_state.model, st.session_state.model_acc = train_model()

    model = st.session_state.model

    #Header
    st.markdown("## 🔍 Suspicious URL Detector")

    if st.session_state.model_acc is not None:
        st.caption(f"Model accuracy on held-out test set: {st.session_state.model_acc:.0%}")
    elif model is not None:
        st.caption("Model trained on all data (dataset too small for an accuracy split).")

    #Retrain button
    if st.button("🔄 Retrain model from CSV"):
        with st.spinner("Retraining…"):
            st.session_state.model, st.session_state.model_acc = train_model()
            model = st.session_state.model
        if model is not None:
            acc_msg = (
                f" Accuracy: {st.session_state.model_acc:.0%}"
                if st.session_state.model_acc is not None
                else " (dataset too small to measure accuracy)"
            )
            st.success(f"Retrained on {len(load_csv_db())} URLs.{acc_msg}")
        else:
            st.error(f"`{CSV_DB_PATH}` appears empty or unreadable. Check the file and retry.")

    st.divider()

    # URL input 
    user_url = st.text_input("Enter a URL to check (with or without https://)")

    if st.button("Check URL") and user_url.strip():
        url = normalize_url(user_url.strip())

        if model is None:
            st.warning(
                f"No training data found in `{CSV_DB_PATH}`. "
                "Make sure the file exists with `url` and `label` columns, then click 🔄 Retrain."
            )
        else:
            label, risk, features = predict_url(model, url)
            in_db = is_in_csv_db(url, label=1)

            color = "🔴" if label == "Suspicious" else "🟢"
            st.markdown(f"### {color} Result: **{label}** — Risk: {risk:.0%}")

            st.markdown("#### URL Analysis")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Length", features["url_length"])
            c2.metric("Dots", features["num_dots"])
            c3.metric("HTTPS", "✅ Yes" if features["is_https"] else "❌ No")
            c4.metric("IP Address", "🌐 Yes" if features["has_ip"] else "No")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("'@' Symbol", "⚠️ Yes" if features["has_at"] else "No")
            c6.metric("Hyphen (-)", "Yes" if features["has_dash"] else "No")
            c7.metric("Keywords", features["suspicious_kws"])
            c8.metric("Subdomains", features["num_subdomains"])

            st.markdown("#### Explanation")
            for reason in explain(features, label, in_db):
                st.markdown(f"- {reason}")

            # Update scan history
            st.session_state.history.insert(0, (url, label, risk))
            st.session_state.history = st.session_state.history[:5]

    #recent scans
    if st.session_state.history:
        st.markdown("### Recent Scans")
        for h_url, h_label, h_risk in st.session_state.history:
            icon = "✅" if h_label == "Safe" else "🚨"
            display = h_url if len(h_url) <= 55 else h_url[:52] + "…"
            st.markdown(f"{icon} {display} — {h_label} ({h_risk:.0%})")


if __name__ == "__main__":
    main()