import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("review_turnaround_hours", axis=1)
y = df["review_turnaround_hours"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("mergegate-review-turnaround-hours")

results = []

def evaluate_model(name, model):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_param("model", name)
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })
        mlflow.set_tag("domain", "code_review")

        mlflow.sklearn.log_model(model, name)

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        }

# Models
lasso = Lasso()
rf = RandomForestRegressor(random_state=42)

results.append(evaluate_model("Lasso", lasso))
results.append(evaluate_model("RandomForest", rf))

# Find best
best = min(results, key=lambda x: x["mae"])

# Save JSON
import json
output = {
    "experiment_name": "mergegate-review-turnaround-hours",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")

import joblib

# Save best model
best_model_name = best["name"]
if best_model_name == "Lasso":
    final_model = lasso
else:
    final_model = rf

joblib.dump(final_model, "models/best_model.pkl")