"""
Train and evaluate five classifiers on the Telco Customer Churn dataset.

Run from the project root:
    python model/train_models.py

Outputs (written to model/ and the project root):
    model/<name>.joblib        - full preprocessing + classifier pipelines
    model/metrics.json         - evaluation metrics on the held-out test split
    model/metrics_table.md     - the same metrics as a Markdown table (for README)
    test_data.csv              - the held-out test split (features + target)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
MODEL_DIR = ROOT / "model"
SEED = 42

TARGET = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS  # 19 input features


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    # TotalCharges is stored as text and is blank for 11 brand-new customers (tenure 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ]
    )


def build_models() -> dict:
    # class_weight="balanced" is used where supported, because only ~26.5% of
    # customers churn and we care about catching churners (recall), not just accuracy.
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "Naive Bayes": GaussianNB(),
        # No probability=True (deprecated in newer scikit-learn); AUC uses decision_function scores.
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def positive_scores(pipe, X):
    """Score for the positive class: probability if available, else SVM decision value."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return pipe.decision_function(X)


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_and_clean(DATA_PATH)
    print(f"Dataset: {df.shape[0]} rows, {len(FEATURES)} features")
    print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3), "\n")

    X = df[FEATURES]
    y = (df[TARGET] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Save the held-out split with the original Yes/No target so it can be uploaded to the app.
    test_df = X_test.copy()
    test_df[TARGET] = np.where(y_test == 1, "Yes", "No")
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    results = {}
    for name, clf in build_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = positive_scores(pipe, X_test)
        results[name] = {k: round(float(v), 4) for k, v in evaluate(y_test, y_pred, y_prob).items()}
        joblib.dump(pipe, MODEL_DIR / file_name(name), compress=3)
        print(f"{name:<24}", results[name])

    meta = {"features": FEATURES, "target": TARGET, "positive_label": "Yes",
            "train_rows": len(X_train), "test_rows": len(X_test), "seed": SEED}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    header = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |\n|---|---|---|---|---|---|---|\n"
    rows = "".join(
        f"| {n} | " + " | ".join(f"{m[k]:.4f}" for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]) + " |\n"
        for n, m in results.items()
    )
    (MODEL_DIR / "metrics_table.md").write_text(header + rows)
    print("\nSaved models, metrics and test_data.csv")


if __name__ == "__main__":
    main()
