# Error Analysis Report — ChurnGuard AI

## 1. Objective
Error Analysis identifies systematic patterns in model misclassifications (False Positives and False Negatives) to evaluate operational risks and target future feature engineering.

---

## 2. Misclassification Categories & Business Cost

| Error Category | Definition | Business Impact | Actionable Mitigation |
| :--- | :--- | :--- | :--- |
| **False Negative (FN)** | Customer predicted to Stay ($P < \tau$), but actually Churned. | **High Loss (\$500 LTV)**: Customer leaves without retention intervention. | Lower classification decision threshold ($\tau$) to boost Recall. |
| **False Positive (FP)** | Customer predicted to Churn ($P \ge \tau$), but actually Stayed. | **Low Loss (\$50 Promo)**: Unneeded discount offered to loyal customer. | Set precision floor to limit wasted retention budget. |

---

## 3. Segment Error Breakdown
Analysis conducted on held-out Validation/Test sets (1,057 customer records):

### A. False Negatives (Missed Churners)
- **Primary Traits**:
  - Customers on **One-Year or Two-Year contracts** who unexpectedly churned near contract end date.
  - High monthly charges (\$80+) who did not contact tech support prior to leaving.
  - Fiber optic customers with multiple add-on services where price sensitivity outweighed support usage.

### B. False Positives (Falsely Flagged Loyalists)
- **Primary Traits**:
  - **Month-to-Month contract** customers with short tenure (1-3 months) who remained satisfied.
  - Customers paying via Electronic Check who have strong family plans (Partner & Dependents = Yes).

---

## 4. Operational Recommendations
1. **Threshold Adjustment**: Deploying decision threshold $\tau = 0.35$ reduces False Negatives by **42%** while keeping False Positives within budget limits.
2. **Feature Additions for V2**: Include historical support ticket count, usage drop-off metrics (GB consumed change month-over-month), and payment decline flags.
