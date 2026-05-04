# ChargeWise MVP

Decision-support dashboard for an EV charging AI system.

The MVP demonstrates a complete working pipeline:

1. Generate synthetic EV charging demand data for Bengaluru-style city zones.
2. Train a simple Random Forest demand model.
3. Forecast the next 24 hours of demand.
4. Detect peak-load risk and suggest off-peak charging shifts.
5. Rank zones for new charging station placement.
6. Serve insights through FastAPI and Streamlit.

## Project Structure

```text
chargewise/
  config.py
  data.py
  model.py
  recommendations.py
  station_placement.py
  pipeline.py
api/
  main.py
dashboard/
  app.py
data/
  synthetic_ev_demand.csv
models/
  demand_model.joblib
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Pipeline

```bash
python -m chargewise.pipeline
```

This creates:

- `data/synthetic_ev_demand.csv`
- `models/demand_model.joblib`

## Run the API

```bash
uvicorn api.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /predict-demand`
- `GET /get-recommendations`
- `GET /get-locations`
- `GET /demo-scenario`

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## Deploy on Streamlit Community Cloud

Use these settings when creating the app:

- Repository: `krishajoshi7/ChargeWise`
- Branch: `main`
- Main file path: `dashboard/app.py`

Streamlit Cloud installs packages from `requirements.txt`. The dashboard generates its demo data
and model automatically on first startup, so no local `data/` or `models/` files need to be uploaded.

## Demo Story

Use the dashboard to show:

1. Current EV demand pattern.
2. Grid risk around the evening peak.
3. Smart schedule recommendation shifting load to late night / early morning.
4. Before vs after peak reduction.
5. Top recommended zones for new charging stations.
