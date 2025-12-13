from fastapi import FastAPI, Request
import uvicorn
import pickle
import json
import pandas as pd
import os
from preprocessing_vars import process_vars

app = FastAPI(title="MLOps")

# -----------------------------
#   Carga de modelo y metadata
# -----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "catb_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

# ------------------------
#   Lógica de predicción
# ------------------------
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


# ------------------------------
#     ENDPOINTS PARA SAGEMAKER
# ------------------------------

# Health check requerido por SageMaker
@app.get("/ping")
def ping():
    return {"status": "ok"}

# Endpoint de inferencia requerido por SageMaker
@app.post("/invocations")
async def invoke(request: Request):
    """
    SageMaker envía un JSON:
      {"inputs": {...registro...}}
    o en batch:
      {"inputs": [ {record1}, {record2} ]}
    """
    payload = await request.json()

    # Aceptamos un solo registro o múltiples
    inputs = payload.get("inputs")

    print("MLOps ver 1.0.0")
    print(inputs)

    if isinstance(inputs, dict):
        # caso: un solo registro
        return predict_record(inputs)

    elif isinstance(inputs, list):
        # caso: múltiples registros
        return [predict_record(rec) for rec in inputs]

    else:
        return {"error": "Formato inválido. Debe contener 'inputs'."}

# ----------------------
# Arranque del servidor
# ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # SageMaker usa PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
