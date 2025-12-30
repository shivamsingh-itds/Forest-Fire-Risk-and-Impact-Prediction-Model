import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df):

    df["Fire"] = df["area"].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=["area"])

    df = pd.get_dummies(df, columns=["month", "day"], drop_first=True)

    X = df.drop(columns=["Fire"])
    y = df["Fire"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
