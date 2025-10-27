# 🧠 German Credit Risk Prediction — Cost-Sensitive ML Pipeline with MLflow

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/MLflow-Tracking-orange" alt="MLflow">
  <img src="https://img.shields.io/badge/Explainability-SHAP-green" alt="SHAP">
  <img src="https://img.shields.io/badge/Notebook-Colab-yellow" alt="Colab">
</p>

> End-to-end machine learning pipeline for **credit risk modeling** on the Statlog (German Credit) dataset — featuring **EDA**, **WOE/IV analysis**, **cost-aware evaluation**, **fairness audits**, and **SHAP explainability**, all instrumented for **MLflow tracking** and reproducibility.

---

## 📘 Project Overview

Financial institutions face a dual challenge: maximizing approvals while minimizing default risk.  
This project demonstrates how to construct a **transparent and interpretable credit-scoring system** that balances predictive power with fairness and cost-sensitivity.

**Objectives:**
- Clean and validate the Statlog German Credit dataset  
- Quantify feature predictive strength via **Information Value (IV)**  
- Engineer ordinal, nominal, and numerical representations  
- Train multiple ML models under **cost-asymmetric conditions**  
- Quantify fairness and group disparity  
- Apply **SHAP explainability** for model interpretation  
- Prepare outputs for **MLflow model registration**

---

## 🧩 Pipeline Architecture

| Phase | Description | Key Outputs |
|:------|:-------------|:-------------|
| **1. Data Integrity & EDA** | Unzip, validate, and audit dataset integrity (duplicates, missing values, outliers, leakage check). | `german_clean_main.csv`, baseline cost metrics |
| **2. WOE/IV Analysis** | Calculate **Weight of Evidence (WOE)** and **Information Value (IV)** for categorical variables. Detect monotonic relationships. | `iv_summary.csv`, `woe_<feature>.csv` |
| **3. Numeric Profiling** | Analyze distributional skewness, correlation (Spearman), and bin-wise bad-rate trends. | `numeric_summary.csv`, `numeric_corr_spearman.csv` |
| **4. Interaction & Fairness Audit** | Explore multivariate patterns (Purpose × Amount, Age × History) and generate **fairness disparity reports** for sensitive attributes. | `fairness_*.csv`, `interaction_*.csv` |
| **5. Cost-Sensitive Thresholding** | Estimate expected misclassification cost using asymmetric penalty (False-Negative ×5). Identify **optimal decision threshold**. | `cost_threshold_curve.csv`, `confusion_cost_optimal.csv` |
| **6. Encoding, Scaling & Split** | Apply **ordinal encoding**, **one-hot encoding**, and **RobustScaler** normalization. Split into train/valid/test sets. | Files under `/model_ready/` |
| **7. Model Training & Evaluation** | Train Logistic Regression, Random Forest, and Gradient Boosting under identical conditions. Evaluate AUC, accuracy, and cost metrics. | `model_results_summary.csv`, ROC plots |
| **8. Explainability (SHAP)** | Generate **global feature importance** and **local force plots** for interpretability. | `shap_feature_importance.csv` |

---

## ⚙️ Tech Stack

| Layer | Tools / Libraries |
|:------|:------------------|
| **Data Handling** | `pandas`, `numpy`, `zipfile`, `os` |
| **Visualization** | `matplotlib`, `seaborn`, `shap` |
| **Modeling** | `scikit-learn` (Logistic, RF, GBM) |
| **Evaluation** | ROC-AUC, Cost curves, Precision-Recall |
| **Explainability** | SHAP (TreeExplainer / LinearExplainer) |
| **Tracking** | MLflow (planned integration for experiments & metrics) |

---

## 📊 Model Metrics

| Model | AUC | Accuracy | Expected Cost | True Positives | False Positives | False Negatives | True Negatives |
|:------|:----|:----------|:---------------|:----------------|:----------------|:----------------|:----------------|
| **Random Forest** | 0.81 | 0.75 | 0.15 | 134 | 42 | 27 | 197 |
| **Gradient Boosting** | 0.80 | 0.74 | 0.17 | 130 | 47 | 30 | 193 |
| **Logistic Regression** | 0.77 | 0.73 | 0.18 | 125 | 51 | 35 | 189 |

> Metrics generated on validation data with **cost ratio (FN:FP = 5:1)**.  
> ROC and cost-threshold curves available in `/eda_outputs/`.

---

## 🔍 Reproducibility

To run this project locally:

1. Clone the repository  
   `git clone https://github.com/<your-username>/germancredit-mlflow-pipeline.git`

2. Navigate to the folder  
   `cd germancredit-mlflow-pipeline`

3. Install dependencies  
   `pip install -r requirements.txt`

4. Run the pipeline  
   `python germancredit_mlflow_pipeline.py`

### Generated Outputs

/eda_outputs/  
├── fairness_*.csv  
├── interaction_*.csv  
├── model_ready/  
├── shap_feature_importance.csv  
└── model_results_summary.csv  

---

<p align="center">
  Built and maintained by <a href="https://ronitshahu.com" target="_blank"><b>Ronit Shahu</b></a> — integrating AI systems, automation pipelines, and explainable financial modeling.
</p>
