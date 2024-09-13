import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import boto3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()

    print("Loading data...")
    data = pd.read_csv(os.path.join(args.data_dir, 'fraud_data.csv'))
    
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Saving model...")
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))

    print("Done!")

if __name__ == "__main__":
    main()