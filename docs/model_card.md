# Model Card — ChurnGuard AI

## 1. Model Details
- **Model Name**: ChurnGuard AI Classification Model
- **Model Version**: 1.0.0
- **Model Type**: Supervised Binary Classification (Logistic Regression Pipeline with ColumnTransformer)
- **Model Framework**: Scikit-Learn 1.3+ / XGBoost / CatBoost
- **Release Date**: August 2026

## 2. Intended Use & Target Users
- **Intended Use**: Decision-support platform to predict customer churn risk probabilities, rank high-risk customers, explain risk drivers via SHAP, and suggest tailored retention campaign actions.
- **Target Users**: Customer Success Managers, Marketing Operations, Retention Teams, Business Analysts.
- **Out-of-Scope Use**: Must NOT be used as the sole automated decision-maker for denying service, credit/financial eligibility decisions, or discriminatory pricing.

## 3. Training & Validation Data
- **Dataset**: IBM Telco Customer Churn dataset (7,043 total customer records).
- **Split Strategy**: Stratified 70% Train (4,930 rows), 15% Validation (1,056 rows), 15% Test (1,057 rows).
- **Class Balance**: 26.54% Churners ($y=1$), 73.46% Retained ($y=0$).
- **Features Used**: 28 total features (20 raw attributes + 8 domain-engineered features including `total_services`, `charge_ratio`, `has_security_service`, `has_support_service`, `payment_electronic_check`).

## 4. Empirical Performance Metrics (Held-Out Test Set)

| Model Candidate | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 0.7348 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 |
| **Logistic Regression** | **0.8040** | **0.6420** | **0.5893** | **0.6145** | **0.8451** | **0.6279** |
| **Random Forest** | 0.7831 | 0.6094 | 0.5071 | 0.5536 | 0.8257 | 0.5930 |
| **XGBoost Classifier** | 0.7794 | 0.5929 | 0.5357 | 0.5629 | 0.8221 | 0.6087 |
| **CatBoost Classifier** | 0.7945 | 0.6275 | 0.5536 | 0.5882 | 0.8457 | 0.6460 |

## 5. Threshold Optimization & Financial Trade-off
- **Default Threshold**: $\tau = 0.50$ yields Recall = 0.5893.
- **Optimized Business Threshold**: $\tau = 0.35$ increases Recall to **> 0.82**, capturing uncaptured churners and minimizing total business loss under a \$500 FN vs \$50 FP cost matrix.

## 6. Caveats & Limitations
- Dataset reflects historical telecom subscriber behavior; performance in other verticals requires retraining.
- SHAP attributions demonstrate model feature importance but do not prove causal interventions.
