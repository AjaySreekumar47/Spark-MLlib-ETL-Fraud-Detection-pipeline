from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

def tune_random_forest(df, label_col="is_fraud"):
    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Random Forest base model
    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features")

    # Param grid
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [20, 50])
                 .addGrid(rf.maxDepth, [5, 10])
                 .build())

    # Evaluator
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")

    # CrossValidator
    cv = CrossValidator(estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    # Pipeline
    pipeline = Pipeline(stages=[cv])

    # Train with CV
    cv_model = pipeline.fit(train_df)

    # Evaluate on test set
    predictions = cv_model.transform(test_df)
    auc = evaluator.evaluate(predictions)

    best_model = cv_model.stages[0].bestModel

    return best_model, auc
