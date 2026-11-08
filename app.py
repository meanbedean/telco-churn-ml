"""
ChurnScope - Telco customer churn model evaluator (Streamlit).

Upload the held-out test CSV, pick one of five pre-trained classifiers and see
its evaluation metrics, confusion matrix, classification report and ROC curve.
Models are loaded from model/*.joblib; nothing is retrained in the app.
"""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
COMPARE_ALL = "Compare all five models"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="ChurnScope", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .hero {background: linear-gradient(120deg, #0f4c5c 0%, #1b7a8c 100%);
             color: #fff; padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem;}
      .hero h1 {color: #fff; margin: 0; font-size: 1.9rem;}
      .hero p {margin: .35rem 0 0 0; opacity: .9;}
      div[data-testid="stMetric"] {background: rgba(27,122,140,.08); border: 1px solid rgba(27,122,140,.25);
             border-radius: 12px; padding: .7rem .9rem;}
    </style>
    <div class="hero">
      <h1>📉 ChurnScope</h1>
      <p>Evaluate five pre-trained classifiers on held-out Telco customer data &mdash;
         will a customer churn (leave) or stay?</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


@st.cache_data
def load_meta():
    with open(MODEL_DIR / "metrics.json") as f:
        return json.load(f)


def prepare(df: pd.DataFrame, features: list, target: str):
    """Apply the same light cleaning used at training time; return X, y (0/1)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError("Uploaded file is missing required columns: " + ", ".join(missing))

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    raw_y = df[target].astype(str).str.strip().str.lower()
    y = raw_y.map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    if y.isna().any():
        raise ValueError(f"Target column '{target}' must contain Yes/No (or 1/0) values only.")
    return df[features], y.astype(int)


def scores_for(model, X):
    """Positive-class score: probability when available, else SVM decision value."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def compute_metrics(y, y_pred, y_score):
    auc = roc_auc_score(y, y_score) if y.nunique() == 2 else np.nan
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def plot_confusion(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", cbar=False, ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc(curves):
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
meta = load_meta()["meta"]
models = load_models()

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + Churn column)", type=["csv"])
    use_sample = st.checkbox("Use bundled test_data.csv instead", value=uploaded is None)
    if SAMPLE_TEST.exists():
        st.download_button("⬇️ Download sample test_data.csv", SAMPLE_TEST.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2 · Model")
    choice = st.selectbox("Choose a classifier", list(MODEL_FILES) + [COMPARE_ALL])

    st.divider()
    st.caption(
        f"Models were trained once on {meta['train_rows']:,} customers "
        f"(80% stratified split, seed {meta['seed']}). The app only runs inference."
    )

# ---------------------------------------------------------------- data
if uploaded is not None and not use_sample:
    raw = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_TEST.exists():
    raw = pd.read_csv(SAMPLE_TEST)
    source = "test_data.csv (bundled)"
else:
    st.info("👈 Upload a test CSV or tick **Use bundled test_data.csv** to begin.")
    st.stop()

try:
    X, y = prepare(raw, meta["features"], meta["target"])
except ValueError as e:
    st.error(str(e))
    with st.expander("Required columns"):
        st.write(meta["features"] + [meta["target"]])
    st.stop()

with st.expander(f"📄 Data preview — {source}", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(X):,}")
    c2.metric("Input features", len(meta["features"]))
    c3.metric("Churn rate", f"{y.mean():.1%}")
    st.dataframe(raw.head(20), width="stretch")

# ---------------------------------------------------------------- single model
if choice != COMPARE_ALL:
    model = models[choice]
    y_pred = model.predict(X)
    y_score = scores_for(model, X)
    m = compute_metrics(y, y_pred, y_score)

    st.subheader(f"Evaluation metrics — {choice}")
    cols = st.columns(6)
    for col, name in zip(cols, METRIC_NAMES):
        col.metric(name, "n/a" if np.isnan(m[name]) else f"{m[name]:.4f}")

    left, mid, right = st.columns([1, 1.1, 1.2])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(y, y_pred, choice))
    with mid:
        st.markdown("**ROC curve**")
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, y_score)
            st.pyplot(plot_roc({choice: (fpr, tpr, m["AUC"])}))
        else:
            st.warning("ROC needs both classes in the uploaded file.")
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y, y_pred, labels=[0, 1], target_names=["Stay (No)", "Churn (Yes)"],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    out = raw.copy()
    out["Predicted_Churn"] = np.where(y_pred == 1, "Yes", "No")
    st.download_button("⬇️ Download predictions as CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[choice].split('.')[0]}.csv", mime="text/csv")

# ---------------------------------------------------------------- comparison
else:
    rows, curves, preds = {}, {}, {}
    for name, model in models.items():
        y_pred = model.predict(X)
        y_score = scores_for(model, X)
        rows[name] = compute_metrics(y, y_pred, y_score)
        preds[name] = y_pred
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, y_score)
            curves[name] = (fpr, tpr, rows[name]["AUC"])

    table = pd.DataFrame(rows).T[METRIC_NAMES]
    st.subheader("Comparison of all five models")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#b9e4dd"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.success(f"Best balanced performer on this data (highest MCC): **{best}** "
               f"— MCC {table.loc[best, 'MCC']:.4f}, F1 {table.loc[best, 'F1']:.4f}")

    # Build all figures first so they can be both shown and downloaded.
    long = table.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
    bar_fig, ax = plt.subplots(figsize=(7.5, 3.8))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model", palette="crest", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_title("Metric comparison", fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    bar_fig.tight_layout()
    roc_fig = plot_roc(curves) if curves else None

    cm_fig, cm_axes = plt.subplots(1, 5, figsize=(18, 3.8))
    for ax_i, (name, y_pred) in zip(cm_axes, preds.items()):
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", cbar=False, ax=ax_i,
                    xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"], annot_kws={"size": 13})
        ax_i.set_title(name, fontsize=11)
        ax_i.set_xlabel("Predicted")
        ax_i.set_ylabel("Actual")
    cm_fig.tight_layout()

    # ---- downloads
    st.markdown("**⬇️ Download comparison**")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Metrics table (CSV)", table.round(4).to_csv(index_label="Model").encode(),
                       file_name="model_comparison_metrics.csv", mime="text/csv")
    d2.download_button("Bar chart (PNG)", fig_to_png(bar_fig),
                       file_name="model_comparison_bar_chart.png", mime="image/png")
    if roc_fig is not None:
        d3.download_button("ROC curves (PNG)", fig_to_png(roc_fig),
                           file_name="model_comparison_roc.png", mime="image/png")
    d4.download_button("Confusion matrices (PNG)", fig_to_png(cm_fig),
                       file_name="model_comparison_confusion_matrices.png", mime="image/png")

    left, right = st.columns([1.4, 1])
    with left:
        st.pyplot(bar_fig)
    with right:
        if roc_fig is not None:
            st.pyplot(roc_fig)

    st.markdown("**Confusion matrices**")
    st.pyplot(cm_fig)

st.caption("Telco Customer Churn dataset (IBM sample data, via Kaggle) · ML Assignment 2 · BITS Pilani WILP")
