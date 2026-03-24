import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/data.csv")

X = df[["feature1", "feature2"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

with mlflow.start_run() as run:

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", accuracy)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)