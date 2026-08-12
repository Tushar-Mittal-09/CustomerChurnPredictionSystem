"""
Exploratory Data Analysis (EDA) Module for ChurnGuard AI.
Generates statistical visualizations and saves figure assets under reports/figures/.
"""
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .cleaning import clean_dataset
from .ingestion import ingest_raw_data
from ..utils.config import get_project_root
from ..utils.logger import get_logger

logger = get_logger("EDA")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def generate_eda_plots(df: pd.DataFrame, output_dir: str = "reports/figures") -> Dict[str, Path]:
    """Generates comprehensive EDA charts and saves high-res images to output_dir.

    Args:
        df: Cleaned Telco DataFrame.
        output_dir: Directory path relative to project root.

    Returns:
        Dict[str, Path]: Dictionary mapping plot names to generated image file paths.
    """
    root = get_project_root()
    fig_dir = root / output_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    generated_files = {}

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts = df["Churn"].value_counts(normalize=True) * 100
    sns.barplot(x=["Retained (0)", "Churned (1)"], y=churn_counts.values, hue=["Retained (0)", "Churned (1)"], palette=["#2ecc71", "#e74c3c"], legend=False, ax=ax)
    ax.set_title("Customer Churn Distribution (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    for i, val in enumerate(churn_counts.values):
        ax.text(i, val + 1, f"{val:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    target_path = fig_dir / "target_distribution.png"
    plt.savefig(target_path, dpi=300)
    plt.close()
    generated_files["target_distribution"] = target_path

    # 2. Tenure Distribution by Churn Status
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df, x="tenure", hue="Churn", common_norm=False, fill=True, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Tenure Distribution by Churn Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tenure (Months)")
    ax.set_ylabel("Density")
    plt.tight_layout()
    tenure_path = fig_dir / "tenure_by_churn.png"
    plt.savefig(tenure_path, dpi=300)
    plt.close()
    generated_files["tenure_by_churn"] = tenure_path

    # 3. Monthly Charges Distribution by Churn
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette=["#2ecc71", "#e74c3c"], legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained (0)", "Churned (1)"])
    ax.set_title("Monthly Charges by Churn Status", fontsize=14, fontweight="bold")
    ax.set_ylabel("Monthly Charges ($)")
    plt.tight_layout()
    mc_path = fig_dir / "monthly_charges_by_churn.png"
    plt.savefig(mc_path, dpi=300)
    plt.close()
    generated_files["monthly_charges_by_churn"] = mc_path

    # 4. Contract Type vs Churn Rate
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
    contract_churn["ChurnRate"] = contract_churn["Churn"] * 100
    sns.barplot(data=contract_churn, x="Contract", y="ChurnRate", hue="Contract", palette="viridis", legend=False, ax=ax)
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    for i, val in enumerate(contract_churn["ChurnRate"]):
        ax.text(i, val + 1, f"{val:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    contract_path = fig_dir / "contract_vs_churn.png"
    plt.savefig(contract_path, dpi=300)
    plt.close()
    generated_files["contract_vs_churn"] = contract_path

    # 5. Payment Method vs Churn Rate
    fig, ax = plt.subplots(figsize=(10, 5))
    pm_churn = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
    pm_churn["ChurnRate"] = pm_churn["Churn"] * 100
    sns.barplot(data=pm_churn, x="PaymentMethod", y="ChurnRate", hue="PaymentMethod", palette="mako", legend=False, ax=ax)
    ax.set_title("Churn Rate by Payment Method", fontsize=14, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    plt.xticks(rotation=15)
    for i, val in enumerate(pm_churn["ChurnRate"]):
        ax.text(i, val + 0.8, f"{val:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    pm_path = fig_dir / "payment_method_vs_churn.png"
    plt.savefig(pm_path, dpi=300)
    plt.close()
    generated_files["payment_method_vs_churn"] = pm_path

    # 6. Correlation Heatmap of Numerical Features
    fig, ax = plt.subplots(figsize=(6, 5))
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set_title("Numerical Features Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    corr_path = fig_dir / "correlation_heatmap.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()
    generated_files["correlation_heatmap"] = corr_path

    logger.info(f"Generated {len(generated_files)} EDA plots in {fig_dir}")
    return generated_files


if __name__ == "__main__":
    raw_df, _ = ingest_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    generate_eda_plots(clean_df)
