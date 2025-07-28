import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_simulated_data(output_path="/content/fraud_detection_pipeline/data", n_samples=500_000):
    os.makedirs(output_path, exist_ok=True)

    # Feature pools
    merchant_categories = ['electronics', 'grocery', 'fashion', 'travel', 'restaurant', 'gas_station']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami']
    device_types = ['mobile', 'web', 'ATM']

    # Simulate data
    data = {
        "transaction_id": np.arange(1, n_samples + 1),
        "user_id": np.random.randint(1000, 10000, size=n_samples),
        "timestamp": [datetime(2023, 1, 1) + timedelta(minutes=random.randint(0, 60 * 24 * 180)) for _ in range(n_samples)],
        "amount": np.round(np.random.exponential(scale=100, size=n_samples), 2),
        "merchant_category": np.random.choice(merchant_categories, size=n_samples),
        "location": np.random.choice(locations, size=n_samples),
        "device_type": np.random.choice(device_types, size=n_samples),
    }

    # Add fraud label (~1% fraud)
    fraud_ratio = 0.01
    data["is_fraud"] = np.where(np.random.rand(n_samples) < fraud_ratio, 1, 0)

    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)

    # Save to CSV
    csv_path = os.path.join(output_path, "simulated_transactions.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ Simulated dataset saved to: {csv_path}")
    return df
