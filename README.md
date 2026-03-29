# CliniqAI — Clinical Intelligence Dashboard

A production-ready AI healthcare dashboard built with Streamlit. No backend required — everything runs in a single service.

**Live Demo:** [cliniappi-vqpjtwimlph5vrazj62p8k.streamlit.app](https://cliniappi-vqpjtwimlph5vrazj62p8k.streamlit.app)

---

## Features

### Tab 1 — Patient Risk Prediction
- Input patient demographics, lab values, and vitals via interactive sliders
- Predicts clinical deterioration risk: **LOW / MODERATE / HIGH**
- Displays confidence probability, gauge chart, and feature importance breakdown
- Model: RandomForest (scikit-learn), trained on synthetic clinical data at startup

### Tab 2 — Clinical Note NLP
- Paste any free-text clinical note
- Extracts structured entities across four categories:
  - Symptoms
  - Conditions
  - Medications
  - Lab Values
- Method: Rule-based lexicon matching (150+ symptoms, 80+ medications, 60+ conditions)
- No external model downloads required

### Tab 3 — Vitals Anomaly Detection
- Upload a CSV or use built-in sample data
- Detects anomalies across Heart Rate, Systolic BP, SpO2, and Temperature
- Method: Rolling z-score + clinical reference range thresholds
- Outputs time-series charts with anomaly markers and a stats summary table

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend & Server | Streamlit |
| Risk Model | scikit-learn (RandomForest) |
| NLP | Pure Python (regex + medical lexicons) |
| Anomaly Detection | NumPy / Pandas (rolling statistics) |
| Charts | Plotly |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
cliniqai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
├── Dockerfile              # For Docker / Railway / Render deployment
└── models/
    ├── __init__.py
    ├── risk_model.py       # RandomForest risk predictor
    ├── nlp_model.py        # Clinical entity extractor
    └── vitals_model.py     # Vitals anomaly detector
```

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/cliniqai.git
cd cliniqai

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

App will be available at `http://localhost:8501`

---

## Deploy

### Streamlit Community Cloud (Recommended — Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Click Deploy — live in ~2 minutes

### Docker
```bash
docker build -t cliniqai .
docker run -p 8080:8080 cliniqai
```

### Render / Railway
- Connect your GitHub repo
- Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Deploy

---

## Requirements

```
streamlit>=1.32.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
plotly>=5.19.0
```

No GPU required. Runs under 512MB RAM. No external model downloads.

---

## Architecture Notes

- **No backend server** — all ML logic runs directly inside the Streamlit process
- **No model downloads** — the RandomForest trains on 2,000 synthetic patients at startup (~2s)
- **No heavy NLP libraries** — replaces spaCy/BERT with curated medical lexicons
- **Auto-redeploy** — every `git push` to `main` triggers a redeploy on Streamlit Cloud

---

## Updating the App

```bash
git add .
git commit -m "describe your change"
git push
```

Streamlit Cloud automatically redeploys within ~1 minute.

---

## Disclaimer

CliniqAI is built for demonstration and educational purposes only. It is not validated for clinical use and should not be used to make real medical decisions.

---

## Author

Diutimaya Mohanty