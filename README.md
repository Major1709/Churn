# 📊 Customer Churn Prediction — Machine Learning Project

## Project Objective
The objective of this project is to **predict customer churn** using machine learning models, with a strong emphasis on **business-oriented decision making** rather than raw accuracy.

Churn prediction is a **binary classification problem**:
- `1` → customer churns
- `0` → customer stays

The key business challenge is to **reduce customer loss (False Negatives)** while keeping **commercial intervention costs (False Positives)** at a reasonable level.

---

## Dataset Overview
- **3,333 customers**
- **11 features** describing customer behavior, billing, and contracts
- Target variable: `Churn`
- Dataset is **imbalanced** (~15% churn)

---

## Exploratory Data Analysis (EDA)
The EDA phase focused on understanding customer behavior and churn patterns:
- Identification of numerical vs categorical variables
- Analysis of class imbalance
- Group statistics (churn vs non-churn)
- Boxplots and distributions

Key insights:
- High number of customer service calls strongly increases churn risk
- High usage and high monthly charges are associated with churn
- Contract renewal and data plans significantly reduce churn probability

---

## Statistical Analysis & Feature Selection
Multiple complementary techniques were used:

- **Correlation analysis** for initial exploration
- **Chi-square tests** for categorical variables (e.g. `DataPlan`, `ContractRenewal`)
- **Mutual Information** to capture non-linear dependencies

Final selected features:
- `CustServCalls`
- `MonthlyCharge`
- `DayMins`
- `ContractRenewal`
- `DataPlan`

> Note:
> `DayMins` was excluded from Logistic Regression to reduce multicollinearity but retained for Random Forest, which benefits from non-linear interactions.

---

## Data Preparation
- Train/Test split: **80% / 20%** with stratification
- **Standardization** applied only to Logistic Regression
- No normalization for tree-based models
- Strict prevention of data leakage

---

## Models Evaluated

### 1️⃣ Logistic Regression (Balanced)
- Linear baseline model
- `class_weight="balanced"`
- Threshold tuning applied to improve churn recall
- Interpretable coefficients

### 2️⃣ Random Forest (Balanced) — Final Model
- Non-linear ensemble model
- Captures feature interactions automatically
- No feature scaling required
- More robust to noise and redundancy

---

## Model Performance Comparison

### 🔹 Random Forest
Confusion Matrix:
```
[[474  96]
 [ 19  78]]
```

- Precision (Churn): **0.448**
- Recall (Churn): **0.804**
- F1-score (Churn): **0.576**
- ROC AUC: **0.845**

---

### 🔹 Logistic Regression
Confusion Matrix:
```
[[363 207]
 [ 16  81]]
```

- Precision (Churn): **0.281**
- Recall (Churn): **0.835**
- F1-score (Churn): **0.421**
- ROC AUC: **0.804**

---

## Business-Oriented Model Selection

Key cost considerations:
- **False Negatives (FN)** → lost customers (high cost)
- **False Positives (FP)** → unnecessary commercial actions (lower cost)

| Model | False Positives | False Negatives |
|---|---:|---:|
| Logistic Regression | 207 | **16** |
| Random Forest | **96** | 19 |

### Final Decision
Although Logistic Regression slightly reduces False Negatives, it generates **more than twice as many False Positives**.

**Random Forest is selected as the final model** because it:
- Reduces False Positives by ~54%
- Maintains high churn recall (~80%)
- Achieves a much higher F1-score
- Provides better overall ranking ability (ROC AUC)

From a business perspective, Random Forest represents a **more cost-effective and operationally realistic solution**.

---

## Model Interpretability (SHAP)
- SHAP values were used to explain Random Forest predictions
- Key churn drivers:
  - Customer service calls
  - Daytime usage
  - Monthly charges
  - Contract renewal status
- SHAP provides both **global feature importance** and **local (customer-level) explanations**

---

## Model Persistence
- Final Random Forest model saved using `joblib`
- Feature list stored to ensure consistent inference
- Model ready for deployment or further optimization

---

## Key Takeaways
- Accuracy alone is misleading in churn prediction
- Recall, precision, and business cost must guide decisions
- Threshold tuning is as important as model choice
- Random Forest outperforms Logistic Regression in real-world churn scenarios
- Interpretability is essential for business adoption

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP
- Joblib

---

📌 **This project demonstrates a complete end-to-end machine learning pipeline with strong business alignment and production-ready considerations.**

