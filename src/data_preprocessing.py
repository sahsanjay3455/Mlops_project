import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess(input_path, output_path):
    df = pd.read_csv("data/raw/titanic.csv")
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    print(df.isnull().sum())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    preprocess("data/raw/titanic.csv", "data/processed/data.csv")
    print("Data preprocessing completed. Processed data saved to data/processed/data.csv")
    


