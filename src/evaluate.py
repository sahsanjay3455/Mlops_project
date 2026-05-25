import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate():
    df = pd.read_csv("data/processed/data.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    model = joblib.load("models/model.pkl")

    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    print(f"Accuracy: {acc}")

if __name__ == "__main__":
    evaluate()
    print("Model evaluation completed.")


