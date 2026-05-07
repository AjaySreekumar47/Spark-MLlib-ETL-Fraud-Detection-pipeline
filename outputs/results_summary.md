# Fraud Detection Pipeline Results

## Model Comparison

| model         |      AUC |   TP |    TN |   FP |   FN |   Precision |    Recall |        F1 |
|:--------------|---------:|-----:|------:|-----:|-----:|------------:|----------:|----------:|
| logistic      | 0.543551 |    0 | 19831 |    0 | 8675 |    0        | 0         | 0         |
| random_forest | 0.603151 |    0 | 19831 |    0 | 8675 |    0        | 0         | 0         |
| gbt           | 0.676876 |  199 | 19777 |   54 | 8476 |    0.786561 | 0.0229395 | 0.0445789 |

## Random Forest Cross-Validation

Best Random Forest cross-validation AUC: `0.9643`

Note: Cross-validation AUC is reported separately from the held-out model comparison table.
