from fastapi import FastAPI

app = FastAPI(title="Diabetes Triage Service")

@app.get("/health")
def health():
    return {"status": "ok"}
