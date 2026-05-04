from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("models/best_model.pkl")

class InputData(BaseModel):
    pr_lines_changed: int = Field(..., ge=10, le=2000)
    reviewer_load: int = Field(..., ge=1, le=15)
    file_count: int = Field(..., ge=1, le=50)
    is_critical_path: int = Field(..., ge=0, le=1)

@app.get("/ping")
def ping():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.pr_lines_changed,
                   data.reviewer_load,
                   data.file_count,
                   data.is_critical_path]])

    prediction = model.predict(X)[0]

    return {"prediction": float(prediction)}