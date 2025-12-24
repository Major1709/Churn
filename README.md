# Churn
📊 Customer Churn Prediction — Machine Learning Project
Project Objective

The objective of this project is to predict customer churn using machine learning models, with a strong focus on business-oriented evaluation rather than raw accuracy.

Churn prediction is a binary classification problem where:

1 = customer churns

0 = customer stays

The primary business goal is to minimize customer loss (False Negatives) while keeping commercial intervention costs (False Positives) under control.

Dataset Overview

3,333 customers

11 features describing usage, billing, and contract behavior

Target variable: Churn

Dataset is imbalanced (~15% churn)

Feature Engineering & Selection

Feature selection was guided by:

Exploratory Data Analysis (EDA)

Correlation analysis

Chi-square statistical tests

Mutual Information

Model behavior and interpretability

Final selected features:

CustServCalls

MonthlyCharge

DayMins

ContractRenewal

DataPlan

Note:
DayMins was excluded from Logistic Regression to reduce multicollinearity but retained for Random Forest, which benefits from non-linear interactions.

Models Evaluated
1️⃣ Logistic Regression (Balanced)

Linear, interpretable baseline model

Uses class_weight="balanced"

Threshold adjusted to prioritize churn detection

Requires feature standardization

2️⃣ Random Forest (Balanced)

Non-linear ensemble model

Captures complex interactions automatically

No feature scaling required

More robust to noise and redundancy

Model Performance Comparison
🔹 Random Forest

Confusion Matrix:

[[474  96]
 [ 19  78]]


Precision (Churn): 0.448

Recall (Churn): 0.804

F1-score (Churn): 0.576

ROC AUC: 0.845

🔹 Logistic Regression

Confusion Matrix:

[[363 207]
 [ 16  81]]


Precision (Churn): 0.281

Recall (Churn): 0.835

F1-score (Churn): 0.421

ROC AUC: 0.804

Business-Oriented Evaluation
Key Cost Considerations

False Negatives (FN) = churn not detected → lost customer

False Positives (FP) = unnecessary commercial action → operational cost

Model	FP	FN
Logistic Regression	207	16
Random Forest	96	19
Final Model Decision

Although Logistic Regression slightly reduces False Negatives (16 vs 19), it generates more than twice as many False Positives, leading to significantly higher commercial costs.

Random Forest is selected as the final model because:

It reduces False Positives by ~54%

Maintains a high churn recall (~80%)

Achieves a much higher F1-score

Provides better overall ranking ability (ROC AUC)

From a business perspective, Random Forest represents a more cost-effective and operationally realistic solution.

Model Interpretability

SHAP values were used to explain Random Forest predictions

Key churn drivers identified:

Customer service calls

Daytime usage

Monthly charges

Contract renewal status

SHAP provides both global insights and customer-level explanations, supporting trust and decision-making

Model Deployment

Final Random Forest model saved using joblib

Feature list stored to ensure consistent inference

Model ready for deployment or further optimization

Key Takeaways

Accuracy alone is misleading in churn prediction

Recall, precision, and business cost must guide model choice

Threshold tuning is as important as algorithm selection

Random Forest outperforms Logistic Regression in real-world churn scenarios

Interpretability (SHAP) is essential for business adoption

Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

SHAP

Joblib

📌 This project demonstrates a complete, end-to-end machine learning pipeline with strong business alignment and production-ready considerations.
