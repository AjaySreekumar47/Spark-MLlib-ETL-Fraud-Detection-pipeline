from pyspark.sql.functions import col
from pyspark.sql import DataFrame

def oversample_minority(df: DataFrame, label_col="is_fraud", target_ratio=0.5) -> DataFrame:
    """
    Oversample the minority class to achieve a target minority-to-total ratio.
    Only Spark-native methods used.
    """
    # Separate majority and minority classes
    majority_df = df.filter(col(label_col) == 0)
    minority_df = df.filter(col(label_col) == 1)

    # Count current class sizes
    majority_count = majority_df.count()
    minority_count = minority_df.count()

    # Calculate how many times to replicate minority class
    desired_minority_count = int(target_ratio * (majority_count + minority_count) / (1 - target_ratio))
    multiplier = desired_minority_count // minority_count

    # Replicate minority class
    oversampled_minority = minority_df
    for _ in range(multiplier - 1):
        oversampled_minority = oversampled_minority.union(minority_df)
    
    # Add remaining rows to meet the target exactly
    remaining = desired_minority_count - oversampled_minority.count()
    if remaining > 0:
        oversampled_minority = oversampled_minority.union(minority_df.limit(remaining))

    # Combine with majority class
    balanced_df = majority_df.union(oversampled_minority)
    return balanced_df
