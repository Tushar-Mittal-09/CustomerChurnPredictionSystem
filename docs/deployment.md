# Deployment & MLOps Monitoring Strategy — ChurnGuard AI

## 1. Containerized Architecture
ChurnGuard AI is fully containerized using Docker and Docker Compose:
- **FastAPI Microservice** (`churnguard_api`): Exposes REST API on port `8000`.
- **Streamlit Web Dashboard** (`churnguard_dashboard`): Exposes interactive UI on port `8501`.

```bash
# Build and launch multi-container application
docker compose up --build
```

---

## 2. Cloud Deployment Options

### A. AWS (ECS / Fargate)
1. Push Docker image to **AWS Elastic Container Registry (ECR)**.
2. Deploy container tasks on **AWS ECS Fargate** behind an Application Load Balancer (ALB).
3. Store logs in **Amazon CloudWatch**.

### B. GCP (Cloud Run)
1. Push container image to **Google Artifact Registry**.
2. Deploy directly via `gcloud run deploy churnguard-api --image ...`.

### C. Render / Railway
1. Connect GitHub repository to Render/Railway.
2. Select Docker build context to deploy API and Dashboard web services.

---

## 3. Production Monitoring & Drift Strategy

### A. Data Drift Monitoring
- **Features Monitored**: `MonthlyCharges`, `tenure`, `Contract`, `TotalCharges`.
- **Metrics**: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test against baseline training distributions.
- **Alert Trigger**: $PSI > 0.25$ triggers automated data drift alert.

### B. Prediction Drift Monitoring
- **Metrics**: Rolling 7-day average predicted churn probability $E[P(\text{Churn})]$ and percentage of customers flagged `HIGH RISK`.
- **Alert Trigger**: Shift of $> 15\%$ in average predicted probability.

### C. Model Retraining Strategy
- **Scheduled Retraining**: Monthly automated retraining pipeline execution.
- **Triggered Retraining**: Triggered when PSI exceeds drift thresholds or when quarterly ground-truth churn outcomes arrive showing F1 degradation $> 5\%$.
