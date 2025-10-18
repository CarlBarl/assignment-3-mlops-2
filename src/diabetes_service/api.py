from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import json

app = FastAPI(title="Diabetes Triage Service")

# ----------- data model (input validation) -----------
class DiabetesFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


# ----------- load model & metadata on startup -----------
MODEL_PATH = Path("artifacts/model.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

_model = None
_rmse = None


@app.on_event("startup")
def load_model():
    global _model, _rmse
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    else:
        raise RuntimeError("Model file not found. Run train.py first!")

    if METRICS_PATH.exists():
        _rmse = json.load(open(METRICS_PATH))["rmse"]
    else:
        _rmse = None


# ----------- endpoints -----------
@app.get("/health")
def health():
    return {"status": "ok", "model_rmse": _rmse}


@app.post("/predict")
def predict(features: DiabetesFeatures):
    try:
        X = [[
            features.age, features.sex, features.bmi, features.bp,
            features.s1, features.s2, features.s3, features.s4,
            features.s5, features.s6
        ]]
        prediction = float(_model.predict(X)[0])
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
