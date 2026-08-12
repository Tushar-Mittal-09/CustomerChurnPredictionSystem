# Interview Preparation Guide — 25 Technical & Business Questions & Answers

## Business Context
### 1. Why did you build this project?
Acquiring new customers in subscription verticals (like Telco) costs 5x to 25x more than retaining existing ones. ChurnGuard AI was built to solve the business problem of identifying at-risk customers early, explaining risk drivers using SHAP, and suggesting actionable retention strategies to maximize Customer Lifetime Value (LTV).

### 2. How would a company use it?
Customer retention teams and success managers use the API or Streamlit dashboard to screen customer risk scores daily. High-risk customers ($P \ge 0.35$) automatically trigger retention workflows (e.g. promotional contract discounts for month-to-month subscribers or dedicated tech support follow-up).

### 3. What is the cost of a false negative?
A False Negative means predicting a customer will Stay when they actually Churn. The business loses the full Customer Lifetime Value (LTV), assumed at \$500 per customer in our cost matrix analysis.

### 4. Why shouldn't the company blindly contact every predicted churner?
Contacting every predicted churner incurs intervention costs (e.g. \$50 promotional discount per customer) and risks "sleeping dog" effects—reminding satisfied customers of their subscription, leading them to cancel.

---

## Data Understanding & Pipeline
### 5. How did you handle missing values?
In the IBM Telco dataset, 11 rows in `TotalCharges` contained whitespace strings (`" "`) because `tenure == 0` (newly onboarded subscribers). We logically imputed `TotalCharges = 0.0` for zero tenure and used median imputation for numerical features via Scikit-Learn `SimpleImputer(strategy="median")`.

### 6. How did you identify data leakage?
We conducted an explicit target leakage audit before feature engineering, ensuring no features utilized post-churn customer actions or target information. `customerID` was explicitly purged from predictive features to prevent non-generalizable memorization.

### 7. Why did you remove customerID?
`customerID` is an arbitrary unique string identifier. Including high-cardinality unique IDs causes overfitting, where models memorize specific customer strings rather than learning generalizable behavioral rules.

### 8. How did you encode categorical features?
We used a Scikit-Learn `ColumnTransformer` applying `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` to handle non-numeric attributes (`Contract`, `PaymentMethod`, `InternetService`).

---

## Machine Learning & Optimization
### 9. Why Logistic Regression?
Logistic Regression served as a strong interpretable linear baseline, providing well-calibrated baseline probability outputs and high training efficiency.

### 10. Why XGBoost?
XGBoost is an optimized gradient boosting framework that excels at capturing non-linear feature interactions and handle imbalanced tabular datasets using internal `scale_pos_weight`.

### 11. Why CatBoost?
CatBoost natively handles categorical structures efficiently, resists overfitting via symmetric decision trees, and achieved our highest PR-AUC score (0.6460).

### 12. Why isn't accuracy sufficient?
Because customer churn is imbalanced (only 26.54% churned). A naive majority-class model predicting "No Churn" for every customer achieves 73.46% accuracy while catching 0% of churners (Recall = 0).

### 13. Why is recall important?
Recall measures the proportion of actual churners correctly identified. High recall ensures the business captures at-risk customers before they cancel.

### 14. What is PR-AUC?
Precision-Recall Area Under Curve (PR-AUC) evaluates model ranking performance specifically on the minority positive class without being inflated by a large number of true negatives.

### 15. How did you handle class imbalance?
We evaluated class weighting (`class_weight="balanced"`) and Bayesian Optuna hyperparameter tuning, combined with business cost threshold optimization.

### 16. Why optimize the threshold?
Default classification assumes threshold $\tau = 0.50$. In asymmetric cost environments (\$500 FN vs \$50 FP), lowering $\tau$ to 0.35 significantly improves Recall and minimizes net financial loss.

### 17. How did you select the final model?
The final model (Logistic Regression / CatBoost) was selected based on held-out test ROC-AUC (0.8451), PR-AUC (0.6279), inference speed, and interpretability.

---

## Explainability (SHAP)
### 18. What is SHAP?
SHAP (SHapley Additive exPlanations) is a game-theoretic approach that calculates the marginal contribution of each feature to an individual customer's churn prediction score.

### 19. Does SHAP prove causality?
No. SHAP measures feature attribution within the machine learning model, not causal intervention effects in the real world.

### 20. How would you explain a prediction to a non-technical stakeholder?
"Customer C10482 has an 87% churn probability primarily because they are on a Month-to-Month contract with high monthly charges (\$89.50) and lack tech support service."

---

## Production & MLOps
### 21. How does FastAPI load the model?
FastAPI loads the serialized Scikit-Learn `Pipeline` joblib artifact into memory at startup using `joblib.load()`, allowing instant inference without disk I/O overhead per request.

### 22. How did you ensure preprocessing consistency?
By wrapping feature imputers, scalers, and encoders inside a single Scikit-Learn `Pipeline`, ensuring identical transformations are applied during training, testing, and production inference.

### 23. Why Docker?
Docker encapsulates Python 3.11, C++ library dependencies (XGBoost/CatBoost), FastAPI, and Streamlit into immutable containers, eliminating "works on my machine" inconsistencies across development and cloud environments.

### 24. How would you monitor the model?
By tracking Population Stability Index (PSI) for data drift on key features (`MonthlyCharges`, `tenure`) and monitoring rolling 7-day average predicted churn probabilities.

### 25. How would you retrain it?
By scheduling automated monthly retraining pipelines using Cloud Composer / Airflow or setting automated triggers when PSI drift exceeds 0.25.
