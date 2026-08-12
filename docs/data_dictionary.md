# Data Dictionary — IBM Telco Customer Churn Dataset

| Feature Name | Description | Raw Data Type | Role | Missing Values / Anomaly | Potential Leakage? | Recommended Transformation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `customerID` | Unique customer identifier | Object (String) | ID Column | None | High (ID) | **Exclude from ML features** |
| `gender` | Customer gender (`Male`, `Female`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `SeniorCitizen` | Whether customer is senior citizen (`0`, `1`) | Integer | Categorical/Binary | None | No | Keep as binary flag |
| `Partner` | Whether customer has a partner (`Yes`, `No`) | Object (String) | Categorical/Binary | None | No | Binary Encoding / One-Hot |
| `Dependents` | Whether customer has dependents (`Yes`, `No`) | Object (String) | Categorical/Binary | None | No | Binary Encoding / One-Hot |
| `tenure` | Number of months customer has stayed with company | Integer | Numeric | None | No | Standard Scaling / Binning |
| `PhoneService` | Whether customer has phone service (`Yes`, `No`) | Object (String) | Categorical/Binary | None | No | Binary Encoding / One-Hot |
| `MultipleLines` | Multiple phone lines (`Yes`, `No`, `No phone service`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `InternetService` | Customer internet provider (`DSL`, `Fiber optic`, `No`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `OnlineSecurity` | Tech service flag (`Yes`, `No`, `No internet service`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `OnlineBackup` | Tech service flag (`Yes`, `No`, `No internet service`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `DeviceProtection` | Tech service flag (`Yes`, `No`, `No internet service`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `TechSupport` | Tech service flag (`Yes`, `No`, `No internet service`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `StreamingTV` | Service flag (`Yes`, `No`, `No internet service`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `StreamingMovies` | Service flag (`Yes`, `No`, `No internet service`) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `Contract` | Contract duration (`Month-to-month`, `One year`, `Two year`) | Object (String) | Categorical | None | No | One-Hot Encoding / Ordinal |
| `PaperlessBilling` | Paperless billing flag (`Yes`, `No`) | Object (String) | Categorical/Binary | None | No | Binary Encoding / One-Hot |
| `PaymentMethod` | Payment option (`Electronic check`, `Mailed check`, etc.) | Object (String) | Categorical | None | No | One-Hot Encoding |
| `MonthlyCharges` | Amount charged to customer monthly | Float | Numeric | None | No | Standard Scaling |
| `TotalCharges` | Total amount charged to customer | Object → Float | Numeric | 11 blank spaces (`" "`) | No | Convert to float; impute 0 for tenure=0 |
| `Churn` | Target variable: Customer churned (`Yes`, `No`) | Object (String) | **Target** | None | Target | Map `Yes` → 1, `No` → 0 |
