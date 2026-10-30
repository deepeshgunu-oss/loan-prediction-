"""
Loan Approval Prediction using Machine Learning
Author     : Shashank Deshmukh
Internship : Naviotech Solutions Pvt Ltd (AI & ML)

Pipeline: load data -> EDA -> feature engineering -> preprocessing pipeline
          -> train 6 models -> cross-validate -> tune -> evaluate -> save -> predict
"""

import json
import os
import urllib.request
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
BLUE, ORANGE = "#2e86ab", "#e4572e"

# ------------------------------------------------------------------ 1. Load
DATA_FILE = "loan.csv"
DATA_URL = "https://raw.githubusercontent.com/emmyli/LoanPredictor/master/data/train.csv"
if not os.path.exists(DATA_FILE):  # download automatically if the file is missing
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)
print(df.head(), "\n")
print("Loan status counts:\n", df["Loan_Status"].value_counts(), "\n")
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0], "\n")
print("Duplicate rows:", df.duplicated().sum(), "\n")
print(df.describe().round(1), "\n")

# ------------------------------------------------------------------ 2. EDA plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
counts = df["Loan_Status"].value_counts()
ax[0].bar(["Approved (Y)", "Rejected (N)"], [counts["Y"], counts["N"]], color=[BLUE, ORANGE])
ax[0].set_title("Loan Status Distribution")
ax[0].set_ylabel("Number of applicants")
for i, v in enumerate([counts["Y"], counts["N"]]):
    ax[0].text(i, v + 6, str(v), ha="center")
ax[1].pie([counts["Y"], counts["N"]], labels=["Approved", "Rejected"], autopct="%1.1f%%",
          colors=[BLUE, ORANGE], startangle=90)
ax[1].set_title("Approval Share")
plt.tight_layout(); plt.savefig("fig_class_distribution.png", dpi=150); plt.close()

mv = missing[missing > 0].sort_values()
plt.figure(figsize=(7, 3.8))
plt.barh(mv.index, mv.values, color=BLUE)
for i, v in enumerate(mv.values):
    plt.text(v + 0.5, i, str(v), va="center")
plt.title("Missing Values per Column"); plt.xlabel("Number of missing entries")
plt.tight_layout(); plt.savefig("fig_missing_values.png", dpi=150); plt.close()

df["approved"] = (df["Loan_Status"] == "Y").astype(int)
cats = ["Credit_History", "Property_Area", "Education", "Married", "Gender", "Self_Employed"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
rates = {}
for a, c in zip(axes.ravel(), cats):
    r = df.groupby(c)["approved"].mean() * 100
    rates[c] = {str(k): round(v, 1) for k, v in r.items()}
    a.bar([str(k) for k in r.index], r.values, color=BLUE)
    a.set_title(f"Approval rate by {c}"); a.set_ylim(0, 100); a.set_ylabel("% approved")
    for i, v in enumerate(r.values):
        a.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("fig_approval_rates.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.histplot(df["ApplicantIncome"], bins=40, color=BLUE, ax=ax[0])
ax[0].set_title("Applicant Income (skewed)")
sns.histplot(np.log1p(df["ApplicantIncome"]), bins=40, color=ORANGE, ax=ax[1])
ax[1].set_title("log(Applicant Income): closer to normal")
plt.tight_layout(); plt.savefig("fig_income_distribution.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 3. Feature engineering
df = df.drop(columns=["Loan_ID", "approved"])
df["Dependents"] = df["Dependents"].replace("3+", "3").astype(float)
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["TotalIncome_log"] = np.log1p(df["TotalIncome"])
df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]          # thousands per month
df["BalanceIncome"] = df["TotalIncome"] - df["EMI"] * 1000     # income left after EMI
y = (df["Loan_Status"] == "Y").astype(int)
X = df.drop(columns=["Loan_Status"])

num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Dependents",
            "TotalIncome_log", "LoanAmount_log", "EMI", "BalanceIncome"]
cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area"]

num_corr = pd.concat([X[num_cols], y.rename("Loan_Approved")], axis=1).corr()
plt.figure(figsize=(8, 6.5))
sns.heatmap(num_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Correlation Heatmap (numeric features)")
plt.tight_layout(); plt.savefig("fig_correlation.png", dpi=150); plt.close()

# ------------------------------------------------------------------ 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Train size:", len(X_train), "| Test size:", len(X_test), "\n")
baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"Baseline (always predict 'Approved'): {baseline:.4f}\n")


def make_pipe(clf):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=11),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                    max_depth=2, random_state=RANDOM_STATE),
}

skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted, preds, scores = [], {}, {}, {}
for name, clf in models.items():
    pipe = make_pipe(clf)
    cv = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    s = pipe.predict_proba(X_test)[:, 1]
    fitted[name], preds[name], scores[name] = pipe, yp, s
    rows.append({
        "Model": name,
        "CV_Accuracy": cv.mean(),
        "Accuracy": accuracy_score(y_test, yp),
        "Precision": precision_score(y_test, yp),
        "Recall": recall_score(y_test, yp),
        "F1": f1_score(y_test, yp),
        "ROC_AUC": roc_auc_score(y_test, s),
    })
    r = rows[-1]
    print(f"{name:20s} cv={r['CV_Accuracy']:.4f} acc={r['Accuracy']:.4f} prec={r['Precision']:.4f} "
          f"rec={r['Recall']:.4f} f1={r['F1']:.4f} auc={r['ROC_AUC']:.4f}")

# ------------------------------------------------------------------ 5. Hyper-parameter tuning
print("\nTuning Random Forest and Logistic Regression with GridSearchCV (5-fold)...")
grids = {
    "Random Forest (tuned)": (RandomForestClassifier(random_state=RANDOM_STATE),
                              {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, 7],
                               "clf__min_samples_leaf": [1, 3, 5]}),
    "Logistic Regression (tuned)": (LogisticRegression(max_iter=1000),
                                    {"clf__C": [0.01, 0.1, 1, 10, 100]}),
}
best_params = {}
for name, (clf, grid) in grids.items():
    gs = GridSearchCV(make_pipe(clf), grid, cv=skf, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    pipe = gs.best_estimator_
    best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    yp = pipe.predict(X_test); s = pipe.predict_proba(X_test)[:, 1]
    fitted[name], preds[name], scores[name] = pipe, yp, s
    rows.append({
        "Model": name, "CV_Accuracy": gs.best_score_,
        "Accuracy": accuracy_score(y_test, yp), "Precision": precision_score(y_test, yp),
        "Recall": recall_score(y_test, yp), "F1": f1_score(y_test, yp),
        "ROC_AUC": roc_auc_score(y_test, s),
    })
    r = rows[-1]
    print(f"{name:28s} best={best_params[name]} cv={r['CV_Accuracy']:.4f} acc={r['Accuracy']:.4f} "
          f"prec={r['Precision']:.4f} rec={r['Recall']:.4f} f1={r['F1']:.4f}")

res_df = pd.DataFrame(rows).sort_values(["CV_Accuracy", "F1"], ascending=False).reset_index(drop=True)
print("\n", res_df.round(4).to_string(index=False), "\n")
res_df.to_csv("model_comparison.csv", index=False)

# ------------------------------------------------------------------ 6. Best model and plots
best_name = res_df.loc[0, "Model"]
best = fitted[best_name]
print("Best model (highest cross-validated accuracy):", best_name, "\n")

plt.figure(figsize=(9, 4.8))
x = np.arange(len(res_df)); w = 0.2
for i, m in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
    plt.bar(x + i * w, res_df[m], w, label=m)
plt.xticks(x + 1.5 * w, res_df["Model"], rotation=18, ha="right", fontsize=8)
plt.ylim(0.4, 1.02); plt.title("Model Comparison (test set)"); plt.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("fig_model_comparison.png", dpi=150); plt.close()

cm = confusion_matrix(y_test, preds[best_name])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title(f"Confusion Matrix - {best_name}", fontsize=10)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("fig_confusion_matrix.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4.8))
for name in ["Logistic Regression", "SVM (RBF)", "Random Forest", "Gradient Boosting", best_name]:
    fpr, tpr, _ = roc_curve(y_test, scores[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, scores[name]):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("fig_roc_curves.png", dpi=150); plt.close()

report = classification_report(y_test, preds[best_name], target_names=["Rejected", "Approved"], digits=4)
print(report)

# feature importance from a Random Forest fitted on the same pipeline
rf_pipe = fitted["Random Forest"]
feat_names = rf_pipe.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=feat_names)
imp.index = [n.replace("num__", "").replace("cat__", "") for n in imp.index]
top = imp.sort_values(ascending=False).head(10)
print("Top Random Forest features:\n", top.round(4), "\n")
plt.figure(figsize=(7.5, 4.5))
top[::-1].plot.barh(color=BLUE)
plt.title("Top 10 Feature Importances (Random Forest)"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig("fig_feature_importance.png", dpi=150); plt.close()

# stability of the best model on the full data
cv_full = cross_val_score(best, X, y, cv=skf, scoring="accuracy")
print("5-fold CV accuracy on full data:", cv_full.round(4), "mean", round(cv_full.mean(), 4), "\n")

# ------------------------------------------------------------------ 7. Save
joblib.dump(best, "loan_model.joblib")
tn, fp, fn, tp = cm.ravel()
summary = {
    "rows": int(len(df)), "approved": int(y.sum()), "rejected": int((1 - y).sum()),
    "train": int(len(X_train)), "test": int(len(X_test)), "baseline": round(float(baseline), 4),
    "best_model": best_name, "best_params": best_params,
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "report": report, "results": res_df.round(4).to_dict(orient="records"),
    "cv_full": cv_full.round(4).tolist(), "cv_full_mean": round(float(cv_full.mean()), 4),
    "top_features": {k: round(float(v), 4) for k, v in top.items()},
    "missing": {k: int(v) for k, v in missing[missing > 0].items()},
    "approval_rates": rates,
}
with open("metrics.json", "w") as f:
    json.dump(summary, f, indent=2)


# ------------------------------------------------------------------ 8. Predict for new applicants
def predict_applicant(applicant: dict):
    """applicant: dict with the raw columns of the dataset (without Loan_ID / Loan_Status)."""
    a = pd.DataFrame([applicant])
    a["Dependents"] = a["Dependents"].replace("3+", "3").astype(float)
    a["TotalIncome"] = a["ApplicantIncome"] + a["CoapplicantIncome"]
    a["TotalIncome_log"] = np.log1p(a["TotalIncome"])
    a["LoanAmount_log"] = np.log1p(a["LoanAmount"])
    a["EMI"] = a["LoanAmount"] / a["Loan_Amount_Term"]
    a["BalanceIncome"] = a["TotalIncome"] - a["EMI"] * 1000
    p = best.predict_proba(a)[0, 1]
    return ("APPROVED" if p >= 0.5 else "REJECTED"), p


samples = [
    ("Graduate, married, good credit, income 6000", dict(Gender="Male", Married="Yes", Dependents="1", Education="Graduate",
        Self_Employed="No", ApplicantIncome=6000, CoapplicantIncome=1500, LoanAmount=130,
        Loan_Amount_Term=360, Credit_History=1.0, Property_Area="Semiurban")),
    ("Same profile but bad credit history", dict(Gender="Male", Married="Yes", Dependents="1", Education="Graduate",
        Self_Employed="No", ApplicantIncome=6000, CoapplicantIncome=1500, LoanAmount=130,
        Loan_Amount_Term=360, Credit_History=0.0, Property_Area="Semiurban")),
    ("Low income, large loan, no credit record", dict(Gender="Female", Married="No", Dependents="0", Education="Not Graduate",
        Self_Employed="Yes", ApplicantIncome=1800, CoapplicantIncome=0, LoanAmount=250,
        Loan_Amount_Term=180, Credit_History=0.0, Property_Area="Rural")),
    ("High income, small loan, good credit", dict(Gender="Female", Married="Yes", Dependents="0", Education="Graduate",
        Self_Employed="No", ApplicantIncome=9000, CoapplicantIncome=3000, LoanAmount=120,
        Loan_Amount_Term=360, Credit_History=1.0, Property_Area="Urban")),
]
print("FINAL PREDICTIONS")
print("-" * 78)
for desc, app in samples:
    label, prob = predict_applicant(app)
    print(f"[{label:9s}] P(approve)={prob:.2f}  {desc}")
