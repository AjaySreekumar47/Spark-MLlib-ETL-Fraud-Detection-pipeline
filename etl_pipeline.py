from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, log1p, to_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import os

def build_etl_pipeline(spark, input_csv_path):
    # Read data
    df = spark.read.csv(input_csv_path, header=True, inferSchema=True)

    # Parse timestamp column
    df = df.withColumn("timestamp", to_timestamp("timestamp"))

    # Feature engineering
    df = (
        df.withColumn("hour_of_day", hour(col("timestamp")))
          .withColumn("day_of_week", dayofweek(col("timestamp")))
          .withColumn("amount_log", log1p(col("amount")))
    )

    # Index categorical columns
    cat_cols = ["merchant_category", "location", "device_type"]
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
        for col in cat_cols
    ]

    # Assembling features
    feature_cols = ["hour_of_day", "day_of_week", "amount_log"] + [f"{col}_index" for col in cat_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Pipeline
    pipeline = Pipeline(stages=indexers + [assembler])
    model = pipeline.fit(df)
    df_transformed = model.transform(df)

    # Select only relevant columns
    final_df = df_transformed.select("features", "is_fraud")

    return final_df
