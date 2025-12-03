from fastapi import FastAPI
import uvicorn
import pickle
import json
import pandas as pd
import os
from preprocessing_vars import process_vars

app = FastAPI(title="MLOps")

# MODEL_DIR = "models/2025-12-01_17-29-26"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "catb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

# -----------
# Inferencia
# -----------
def predict_record(record: dict):
    df = pd.DataFrame([record])

    df = process_vars(df)

    # Alinear columnas
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df_model = df[feature_names]

    pred_proba = model.predict_proba(df_model)[0, 1]
    pred_class = int(pred_proba >= 0.5)

    return {
        "pred_class": pred_class,
        "pred_proba": float(pred_proba)
    }

# -------------
# Endpoint API
# -------------
@app.post("/predict")
async def predict_endpoint(payload: dict):
    try:
        result = predict_record(payload)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}

# Asynchronous Server Gateway Interface
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
