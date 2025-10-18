# Virtual Diabetes Clinic Triage – A3 (MLOps)

This project implements an **ML service** that predicts short-term diabetes progression
using scikit-learn’s open diabetes regression dataset.
The service exposes two FastAPI endpoints (`/health` and `/predict`)
and is packaged as a **self-contained Docker image**.

---

## 🏥 Case Overview
Nurses at a virtual diabetes clinic manually review patient check-ins.
This ML service predicts a **progression risk score** so nurses can prioritize follow-up calls.

- **Data:** `load_diabetes()` from `scikit-learn`
- **Target (`y`):** “progression index” (higher = higher risk)

---

## 🚀 How to run locally

### 1️⃣ Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
