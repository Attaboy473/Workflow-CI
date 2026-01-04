import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "train_preprocessed.csv")

# Load dataset
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Fallback jika file ada di root folder (satu tingkat di atas)
    df = pd.read_csv(os.path.join(base_path, "..", "train_preprocessed.csv"))

TARGET_COL = "price_range"
# ... sisa kode Anda tetap sama ...

TARGET_COL = "price_range"

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc}")


