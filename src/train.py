import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow

def train_model():
    df = pd.read_csv("data/processed/data.csv")

    
    X = df.drop(["Survived", "PassengerId"], axis=1)
    y = df["Survived"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)

        joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train_model()
    print("Model training completed and saved to models/model.pkl")
    


