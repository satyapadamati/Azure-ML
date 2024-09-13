import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, weights=[0.97, 0.03], random_state=42)

# Create a DataFrame
columns = [f'feature_{i}' for i in range(20)]
df = pd.DataFrame(X, columns=columns)
df['is_fraud'] = y

# Save to CSV
df.to_csv('../data/fraud_data.csv', index=False)

print("Synthetic data generated and saved to '../data/fraud_data.csv'")