# ChurnScope — Telco Customer Churn Classification

ML Assignment 2 · M.Tech (AIML) · BITS Pilani WILP

**Live app:** `https://telco-churn-2026ak05450.streamlit.app/`

---

## a. Problem statement

A telecom provider loses revenue every time a subscriber cancels service ("churns"). Retaining an existing customer is far cheaper than acquiring a new one, so the business wants to flag at-risk customers early enough to act (discounts, contract upgrades, support calls).

The task is a **binary classification** problem: given a customer's account, demographic and service-subscription details, predict whether the customer will churn (`Churn = Yes`) or stay (`Churn = No`). Five classifiers are trained on the same data, evaluated with six metrics on a held-out test set, and served through an interactive Streamlit app.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Telco Customer Churn (IBM sample dataset) |
| Source | Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| Instances | 7,043 customers |
| Input features | 19 (after dropping the `customerID` identifier) |
| Target | `Churn` — Yes / No |
| Class balance | No 73.5 % · Yes 26.5 % (imbalanced) |

**Features**

- **Numeric (3):** `tenure` (months with the company), `MonthlyCharges`, `TotalCharges`
- **Demographic (4):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services (9):** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account (3):** `Contract`, `PaperlessBilling`, `PaymentMethod`

**Preprocessing**

- `TotalCharges` is stored as text and is blank for 11 customers with `tenure = 0`; it is converted to numeric and the blanks set to 0.
- `SeniorCitizen` (0/1) is mapped to No/Yes so it is treated as categorical.
- Numeric columns are standardised (`StandardScaler`); categorical columns are one-hot encoded (`OneHotEncoder`, unknown categories ignored).
- Data is split 80 / 20 with stratification on the target (`random_state = 42`): 5,634 training rows, 1,409 test rows. The test split is saved as `test_data.csv`.
- Preprocessing and classifier are saved together as one scikit-learn `Pipeline` per model, so the app can take the raw test CSV directly.

## c. GitHub repository link

`https://github.com/meanbedean/telco-churn-ml`

```
telco-churn-ml/
├── app.py                     # Streamlit app (inference only, no retraining)
├── requirements.txt
├── README.md
├── test_data.csv              # held-out 20% test split (19 features + Churn)
├── data/
│   └── telco_customer_churn.csv   # full raw dataset
├── model/
│   ├── train_models.py        # training + evaluation script
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   ├── metrics.json           # metrics + metadata read by the app
│   └── metrics_table.md
└── .streamlit/config.toml     # app theme
```

## d. Models used

All five models use the same preprocessing pipeline and the same train/test split.

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=6`, `min_samples_leaf=20`, `class_weight="balanced"` |
| K-Nearest Neighbor | `n_neighbors=25`, `weights="distance"` |
| Naive Bayes | `GaussianNB()` |
| Support Vector Machine | RBF kernel, `C=1.0`, `class_weight="balanced"` (AUC from `decision_function`) |

`class_weight="balanced"` is used where the algorithm supports it because only about a quarter of customers churn, and missing a churner is the costly error.

### Comparison table (test set, 1,409 customers)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7381 | 0.8415 | 0.5043 | 0.7834 | 0.6136 | 0.4531 |
| Decision Tree | 0.7395 | 0.8336 | 0.5060 | 0.7914 | 0.6173 | 0.4590 |
| KNN | 0.7750 | 0.8087 | 0.5817 | 0.5428 | 0.5615 | 0.4109 |
| Naive Bayes | 0.6955 | 0.8075 | 0.4597 | 0.8396 | 0.5941 | 0.4268 |
| Support Vector Machine | 0.7452 | 0.8213 | 0.5133 | 0.7727 | 0.6169 | 0.4580 |

Precision, Recall and F1 are reported for the positive class (`Churn = Yes`).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.8415), so it ranks churners above non-churners better than any other model. With balanced class weights it catches 78% of churners. Its strong showing says the churn signal is largely linear in the encoded features — month-to-month contracts, short tenure, fibre-optic internet and electronic-cheque payment all push the log-odds up in a fairly additive way. Fast, stable and its coefficients are easy to explain. |
| Decision Tree | Best MCC (0.4590) and F1 (0.6173) and second-best recall (0.79), only marginally ahead of SVM and Logistic Regression. Limiting depth to 6 and requiring 20 samples per leaf kept it from overfitting; an unconstrained tree dropped to MCC 0.30 and AUC 0.65 on the same test set. Its splits on contract type, tenure and internet service give human-readable retention rules. AUC (0.834) is a little below Logistic Regression because a shallow tree outputs only a few distinct probability levels. |
| KNN | Highest accuracy (0.775) and precision (0.58) but the lowest recall (0.54), F1 and MCC. KNN has no class-weight option, so neighbourhoods are dominated by the majority "stay" class and it misses almost half of the churners. This is a clear example of why accuracy is misleading on an imbalanced dataset. One-hot columns also make the distance metric less informative. |
| Naive Bayes | Highest recall (0.84) but the lowest accuracy (0.6955) and precision (0.46): it over-predicts churn. The feature-independence assumption is badly violated (the one-hot service columns are correlated with each other, and `TotalCharges` ≈ `tenure` × `MonthlyCharges`), and Gaussian likelihoods are a poor fit for 0/1 dummy columns. Still a useful cheap baseline with reasonable AUC (0.81). |
| Support Vector Machine | Essentially tied with the Decision Tree on F1 (0.6169) and MCC (0.4580), with good recall (0.77). The RBF kernel did not beat the linear Logistic Regression on AUC (0.821 vs 0.842), again suggesting the decision boundary is close to linear. It is the slowest model to train and does not give native probabilities. |
| **Overall winner for this dataset** | **Decision Tree** has the best MCC and F1, the most balanced measures on this imbalanced problem, while Logistic Regression is a very close second and has the best AUC. The gap between them (≈0.006 MCC) is small, so in practice Logistic Regression is an equally good choice when ranked churn scores or explainable coefficients matter more than a single yes/no threshold. KNN and Naive Bayes are clearly weaker. |

## Streamlit app features

- **CSV upload** of test data (plus a one-click option to use the bundled `test_data.csv`, and a download button for it)
- **Model selection dropdown** — any of the five models, or *Compare all five models*
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC as metric cards
- **Confusion matrix**, **classification report** and **ROC curve** for the selected model
- **Comparison view** — metrics table with best values highlighted, grouped bar chart, overlaid ROC curves and all five confusion matrices
- Downloads: row-level predictions (CSV) for a single model; in the comparison view, the metrics table (CSV) and the bar chart, ROC curves and confusion matrices (PNG)

The app loads the saved `.joblib` pipelines and only runs inference on the uploaded file; it never retrains.

The uploaded CSV must contain the 19 feature columns listed above and the `Churn` column (Yes/No or 1/0).

## How to run locally

```bash
pip install -r requirements.txt
python model/train_models.py     # optional: re-trains and rewrites models, metrics, test_data.csv
streamlit run app.py
```

> The `scikit-learn` version in `requirements.txt` must match the version the models were saved with. If you retrain with a different version, update the pin to that version before deploying.
