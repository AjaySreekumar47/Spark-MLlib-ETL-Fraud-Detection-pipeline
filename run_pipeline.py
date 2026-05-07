import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

from data_simulation import generate_simulated_data
from etl_pipeline import build_etl_pipeline
from imbalance_handler import oversample_minority
from mlflow_logging import log_mlflow_run
from model_training import train_and_evaluate
from rf_hyperparam_tuning import tune_random_forest


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)



def evaluate_thresholds(predictions, label_col="is_fraud", thresholds=None):
    """
    Evaluate precision, recall, and F1 across probability thresholds.
    Assumes predictions contains a Spark MLlib probability column.
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    scored = predictions.withColumn(
        "fraud_probability",
        vector_to_array(col("probability"))[1],
    )

    rows = []

    for threshold in thresholds:
        pred_df = scored.withColumn(
            "threshold_prediction",
            (col("fraud_probability") >= threshold).cast("double"),
        )

        tp = pred_df.filter((col("threshold_prediction") == 1) & (col(label_col) == 1)).count()
        tn = pred_df.filter((col("threshold_prediction") == 0) & (col(label_col) == 0)).count()
        fp = pred_df.filter((col("threshold_prediction") == 1) & (col(label_col) == 0)).count()
        fn = pred_df.filter((col("threshold_prediction") == 0) & (col(label_col) == 1)).count()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        rows.append(
            {
                "threshold": threshold,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    return rows


def save_metrics(metrics_by_model: dict, rf_cv_auc: float | None = None) -> None:
    """Save metrics as JSON, CSV, and Markdown summary."""
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": metrics_by_model,
                "random_forest_cv_auc": rf_cv_auc,
            },
            f,
            indent=2,
        )

    rows = []
    for model_name, metrics in metrics_by_model.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    summary_lines = [
        "# Fraud Detection Pipeline Results",
        "",
        "## Model Comparison",
        "",
        df.to_markdown(index=False),
        "",
    ]

    if rf_cv_auc is not None:
        summary_lines.extend(
            [
                "## Random Forest Cross-Validation",
                "",
                f"Best Random Forest cross-validation AUC: `{rf_cv_auc:.4f}`",
                "",
                "Note: Cross-validation AUC is reported separately from the held-out model comparison table.",
                "",
            ]
        )

    (OUTPUT_DIR / "results_summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def plot_model_comparison(metrics_by_model: dict) -> None:
    """Plot AUC, precision, recall, and F1 for each model."""
    df = pd.DataFrame(
        [
            {
                "model": model_name,
                "AUC": metrics.get("AUC", 0),
                "Precision": metrics.get("Precision", 0),
                "Recall": metrics.get("Recall", 0),
                "F1": metrics.get("F1", 0),
            }
            for model_name, metrics in metrics_by_model.items()
        ]
    )

    ax = df.set_index("model")[["AUC", "Precision", "Recall", "F1"]].plot(
        kind="bar",
        figsize=(10, 6),
    )
    ax.set_title("Spark MLlib Fraud Detection Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close()


def plot_confusion_matrix(metrics: dict, model_name: str) -> None:
    """Plot confusion matrix for one model."""
    matrix = [
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]],
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)

    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close()

def plot_threshold_confusion_matrix(threshold_rows: list[dict], threshold: float = 0.30) -> None:
    """Plot confusion matrix for a selected probability threshold."""
    selected = None
    for row in threshold_rows:
        if abs(row["threshold"] - threshold) < 1e-9:
            selected = row
            break

    if selected is None:
        raise ValueError(f"Threshold {threshold} not found in threshold analysis rows.")

    matrix = [
        [selected["TN"], selected["FP"]],
        [selected["FN"], selected["TP"]],
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix)

    ax.set_title(f"GBT Confusion Matrix at Threshold {threshold:.2f}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_gbt_threshold_{int(threshold * 100)}.png", dpi=160)
    plt.close()



def plot_threshold_analysis(threshold_rows: list[dict]) -> None:
    """Plot precision/recall/F1 across fraud-probability thresholds."""
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(OUTPUT_DIR / "gbt_threshold_analysis.csv", index=False)

    ax = threshold_df.set_index("threshold")[["Precision", "Recall", "F1"]].plot(
        kind="line",
        marker="o",
        figsize=(8, 5),
    )
    ax.set_title("GBT Fraud Threshold Analysis")
    ax.set_xlabel("Fraud Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gbt_threshold_analysis.png", dpi=160)
    plt.close()


def main():
    print("Starting Spark fraud detection pipeline...")

    spark = (
        SparkSession.builder
        .appName("SparkFraudDetectionPipeline")
        .master("local[*]")
        .getOrCreate()
    )

    csv_path = DATA_DIR / "simulated_transactions.csv"

    if not csv_path.exists():
        print("Generating synthetic fraud transaction data...")
        generate_simulated_data(output_path=str(DATA_DIR), n_samples=100_000)
    else:
        print(f"Using existing dataset: {csv_path}")

    print("Building PySpark ETL pipeline...")
    df = build_etl_pipeline(spark, str(csv_path))

    print("Applying Spark-native minority oversampling...")
    balanced_df = oversample_minority(df, label_col="is_fraud", target_ratio=0.3)

    metrics_by_model = {}
    gbt_predictions = None

    for model_type in ["logistic", "random_forest", "gbt"]:
        print(f"Training model: {model_type}")
        _, metrics, predictions = train_and_evaluate(balanced_df, model_type=model_type)

        metrics_by_model[model_type] = metrics
        log_mlflow_run(run_name=f"{model_type}_run", model_type=model_type, metrics=metrics)

        if model_type == "gbt":
            gbt_predictions = predictions

        print(model_type, metrics)

    print("Running Random Forest cross-validation...")
    _, rf_cv_auc = tune_random_forest(balanced_df)
    print(f"Random Forest CV AUC: {rf_cv_auc}")

    save_metrics(metrics_by_model, rf_cv_auc=rf_cv_auc)
    plot_model_comparison(metrics_by_model)
    plot_confusion_matrix(metrics_by_model["gbt"], "gbt")

    if gbt_predictions is not None:
        print("Evaluating GBT probability thresholds...")
        threshold_rows = evaluate_thresholds(gbt_predictions)
        plot_threshold_analysis(threshold_rows)
        plot_threshold_confusion_matrix(threshold_rows, threshold=0.30)
        print(pd.DataFrame(threshold_rows))

    print(f"Saved outputs to: {OUTPUT_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()