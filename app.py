from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        expected_cols = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0   # default value

        df = df[expected_cols]

        prediction = model.predict(df)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}




