# Loan Approval Prediction using Machine Learning

Predicts whether a loan application will be **approved** or **rejected** from the applicant's details, using a scikit-learn preprocessing pipeline and six machine learning models.

**Author:** Deepesh Tahlani
**Internship:** AI & ML, Naviotech Solutions Pvt Ltd (Guide: Mr. Md. Ameen)
**University:** Shri Mahaveer College

## Results

Dataset: Loan Prediction, 614 applicants (422 approved, 192 rejected), 80/20 stratified split (491 train, 123 test). Baseline (always predict "approved"): 69.11%.

| Model | CV accuracy | Test accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Logistic Regression (tuned)** | 79.83% | 86.18% | 84.00% | 98.82% | 90.81% |
| Random Forest (tuned) | 79.83% | 85.37% | 83.17% | 98.82% | 90.32% |
| Logistic Regression | 79.62% | 85.37% | 83.84% | 97.65% | 90.22% |
| Random Forest | 79.42% | 83.74% | 82.83% | 96.47% | 89.13% |
| SVM (RBF) | 79.02% | 84.55% | 82.35% | 98.82% | 89.84% |
| Gradient Boosting | 79.01% | 82.93% | 82.65% | 95.29% | 88.52% |
| Decision Tree | 77.39% | 84.55% | 83.67% | 96.47% | 89.62% |
| KNN | 73.11% | 82.11% | 80.00% | 98.82% | 88.42% |

The test set is small (123 applicants), so its score is noisy. 5-fold cross-validation on all 614 records gives a mean accuracy of **80.30%**, which is the more cautious estimate. The best model was chosen by cross-validated accuracy.

**Key finding:** credit history is by far the most important feature (79.6% approval with good credit history vs 7.9% without). The model rarely rejects good applicants but approves some risky ones (rejected-class recall 57.9%), so a lender would raise the decision threshold.

## Project structure

```
Loan_Prediction_Project/
├── loan_prediction.py                   # full pipeline as a script
├── Loan_Prediction.ipynb                # same pipeline, step by step with outputs
├── loan.csv                             # dataset (downloaded automatically if missing)
├── loan_model.joblib                    # saved trained model
├── requirements.txt
├── Loan_Prediction_Report.docx/.pdf
├── Loan_Prediction_Presentation.pptx
├── screenshots/                         # dataset, training, accuracy, prediction
└── figures/                             # charts used in the report
```

## Setup

Requires Python 3.9 or higher.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install the libraries
python -m pip install -r requirements.txt
```

## Run

**Script**

```bash
python loan_prediction.py
```

It trains and compares all models, runs GridSearchCV, prints the metrics, saves the charts (`fig_*.png`), `model_comparison.csv`, `metrics.json` and `loan_model.joblib`, and predicts four sample applicants. The first run takes under a minute.

**Notebook:** open `Loan_Prediction.ipynb` in Jupyter, or upload it to [Google Colab](https://colab.research.google.com) (no installation needed) and choose Runtime > Run all.

## Predict for a new applicant

The saved pipeline expects the engineered columns, so build them first:

```python
import joblib, numpy as np, pandas as pd

model = joblib.load("loan_model.joblib")
a = pd.DataFrame([dict(Gender="Male", Married="Yes", Dependents="1", Education="Graduate",
                       Self_Employed="No", ApplicantIncome=6000, CoapplicantIncome=1500,
                       LoanAmount=130, Loan_Amount_Term=360, Credit_History=1.0,
                       Property_Area="Semiurban")])
a["Dependents"] = a["Dependents"].replace("3+", "3").astype(float)
a["TotalIncome"] = a["ApplicantIncome"] + a["CoapplicantIncome"]
a["TotalIncome_log"] = np.log1p(a["TotalIncome"])
a["LoanAmount_log"] = np.log1p(a["LoanAmount"])
a["EMI"] = a["LoanAmount"] / a["Loan_Amount_Term"]
a["BalanceIncome"] = a["TotalIncome"] - a["EMI"] * 1000

p = model.predict_proba(a)[0, 1]
print("APPROVED" if p >= 0.5 else "REJECTED", round(p, 2))
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` (or pandas, seaborn, joblib...) | Run `python -m pip install -r requirements.txt` using the same Python that runs the script |
| `pip` not found | Use `python -m pip install -r requirements.txt` |
| `FileNotFoundError: loan.csv` | Run the script from inside the project folder; it also downloads the file automatically if internet is available |
| VS Code uses a different Python | Press Ctrl+Shift+P, choose "Python: Select Interpreter", pick the Python where you installed the libraries |
| Plot windows do not open | Normal: the script saves charts as PNG files instead of showing them |
| Accuracy differs by a point or two | Different scikit-learn versions can change results slightly; the random seed is fixed at 42 |

## Dataset

Loan Prediction practice problem (Dream Housing Finance data) from Analytics Vidhya. The copy used here: https://github.com/emmyli/LoanPredictor (data/train.csv).

## License

Released under the MIT License. See the `LICENSE` file.
