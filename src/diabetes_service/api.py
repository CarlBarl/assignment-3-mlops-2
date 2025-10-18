from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Diabetes Triage Service")


# --- existing health endpoint ---
@app.get("/health")
def health():
    return {"status": "ok"}


# --- new model class describing input data ---
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


# --- new /predict endpoint ---
@app.post("/predict")
def predict(features: DiabetesFeatures):
    try:
        # we’ll replace this dummy value with a real model prediction later
        score = 42.0
        return {"prediction": score}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
