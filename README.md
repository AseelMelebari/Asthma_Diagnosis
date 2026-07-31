# Asthma Risk Prediction API

An educational machine-learning deployment built with FastAPI and scikit-learn. It is intended for testing AI ethics, governance, robustness, and explainability tools.

> **Medical disclaimer:** This project is not a medical device and must not be used for diagnosis, treatment, emergency decisions, or other clinical purposes.

## Files

- `main.py` — FastAPI application
- `asthma_model.joblib` — trained scikit-learn pipeline
- `requirements.txt` — Python dependencies
- `railway.json` — Railway deployment configuration
- `runtime.txt` — Python runtime version
- `static/index.html` — simple browser interface

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

- App: `http://127.0.0.1:8000`
- Swagger API: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Model information: `http://127.0.0.1:8000/model-info`

## API request

`POST /predict`

```json
{
  "age": 35,
  "gender": "Male",
  "smoking_status": "Non-Smoker",
  "medication": "Inhaler",
  "peak_flow": 420
}
```

The categorical text values should use the same spelling found in the training dataset. Unknown categories are accepted by the model's encoder but may reduce prediction reliability.

## Deploy on Railway

1. Upload all files in this folder to the root of the GitHub repository.
2. In Railway, create a new project and select **Deploy from GitHub repo**.
3. Select this repository and deploy it.
4. After deployment succeeds, open **Settings → Networking → Generate Domain**.
5. Open the generated domain or append `/docs` for Swagger.

Railway uses the start command in `railway.json`:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Governance notes

Known risks and limitations:

- The model was trained on a small dataset and may not generalize.
- `Medication` may create target leakage because medication can be prescribed after diagnosis.
- Group fairness has not been established.
- The model has not been clinically validated.
- Do not submit identifiable or real patient data to a public deployment.

Recommended evaluations include fairness, robustness, privacy, explainability, transparency, data quality, and target leakage analysis.
