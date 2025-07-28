# 🚀 Spark-Based Fraud Detection Pipeline

This project demonstrates a scalable, end-to-end **fraud detection system** using distributed data processing and machine learning. It covers PySpark-based ETL, class imbalance handling, model training (in both Colab and Databricks), and interpretable visualizations via SHAP and Streamlit.

---

## 🧠 Project Overview

- **Objective**: Predict fraudulent financial transactions from large-scale simulated data
- **Key Components**:
  - Synthetic data generation (1M+ records)
  - PySpark-based ETL and feature engineering
  - Model training with:
    - `XGBoost` in Google Colab
    - `Spark MLlib` in Databricks
  - Class imbalance handling (via SMOTE/ADASYN in Colab)
  - Evaluation: AUC, precision, recall, F1
  - Interpretability: SHAP plots
  - Optional: Streamlit dashboard, MLflow logging

---

## 🔧 Tech Stack

| Layer             | Tools & Frameworks                                 |
|------------------|-----------------------------------------------------|
| Data Simulation   | PySpark, Pandas                                     |
| ETL & Feature Eng | PySpark (`withColumn`, `log1p`, `hour`, etc.)      |
| Modeling (Colab)  | XGBoost, SMOTE, SHAP                                |
| Modeling (DB)     | Spark MLlib (Logistic Regression)                   |
| Visualization     | SHAP, Seaborn, Matplotlib                           |
| Experimentation   | MLflow (optional)                                   |
| Deployment (opt.) | Streamlit + Cloudflare Tunnel                       |

---

## 🧪 Workflow Summary

### 📍 Colab (GPU)
- ✅ Simulate and preprocess data using PySpark
- ✅ Train models with XGBoost (SMOTE/ADASYN to balance)
- ✅ Evaluate and visualize with SHAP
- ✅ Build interactive dashboard with Streamlit
- ✅ Log experiments to MLflow

### 📍 Databricks (Spark Cluster)
- ✅ Import data into DBFS and load via Spark
- ✅ Repeat ETL + feature engineering
- ✅ Train model with Spark MLlib (`LogisticRegression`)
- ✅ Evaluate using AUC and confusion matrix

> ⚠️ Due to Spark Connect limitations, full MLlib support was only possible in a paid Databricks cluster.

---

## 📊 Sample Outputs

- Confusion matrix comparing multiple classifiers
- SHAP summary plot (top 10 contributing features)
- Streamlit dashboard for live predictions (optional)

---

## 📁 Folder Structure (To Be Added)

Will be tailored after repository name is provided. Includes:
- `data/`
- `notebooks/colab/`
- `notebooks/databricks/`
- `scripts/`
- `streamlit_app/`
- `outputs/` (metrics, plots)
- `README.md`

---

## 📌 Authors & Credits

Created by **Ajay Sreekumar**  
Developed as part of a domain-focused skill-building project in the Data Science and Machine Learning space.

---

## 🏁 Future Enhancements

- Full deployment via Docker or Streamlit Sharing
- Real-world data integration (e.g., Kaggle credit card fraud)
- Model registry with MLflow + CI/CD
