from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import DataFrame

def train_and_evaluate(df: DataFrame, label_col="is_fraud", model_type="logistic"):
    """
    Train a classifier on the input DataFrame and return the model and evaluation metrics.
    Supported model_type: "logistic", "random_forest", "gbt"
    """
    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Select model
    if model_type == "logistic":
        model = LogisticRegression(labelCol=label_col, featuresCol="features")
    elif model_type == "random_forest":
        model = RandomForestClassifier(labelCol=label_col, featuresCol="features", numTrees=50)
    elif model_type == "gbt":
        model = GBTClassifier(labelCol=label_col, featuresCol="features", maxIter=10, maxDepth=5)
    else:
        raise ValueError("Invalid model_type. Choose from: logistic, random_forest, gbt.")

    # Train model
    pipeline = Pipeline(stages=[model])
    model_fitted = pipeline.fit(train_df)

    # Predict on test set
    predictions = model_fitted.transform(test_df)

    # Evaluate
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)
    auc = evaluator.evaluate(predictions)

    # Confusion matrix
    pred_labels = predictions.select(col("prediction"), col(label_col))
    tp = pred_labels.filter((col("prediction") == 1) & (col(label_col) == 1)).count()
    tn = pred_labels.filter((col("prediction") == 0) & (col(label_col) == 0)).count()
    fp = pred_labels.filter((col("prediction") == 1) & (col(label_col) == 0)).count()
    fn = pred_labels.filter((col("prediction") == 0) & (col(label_col) == 1)).count()

    metrics = {
        "AUC": auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "Recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }

    return model_fitted, metrics
