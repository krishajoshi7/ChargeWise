from __future__ import annotations

from fastapi import FastAPI

from chargewise.pipeline import run_pipeline

app = FastAPI(title="ChargeWise EV Charging MVP", version="0.1.0")


def _result():
    return run_pipeline(train=False)


@app.get("/health")
def health():
    return {"status": "ok", "service": "chargewise"}


@app.get("/predict-demand")
def predict_demand():
    predictions = _result()["predictions"].copy()
    predictions["timestamp"] = predictions["timestamp"].astype(str)
    return predictions.to_dict(orient="records")


@app.get("/get-recommendations")
def get_recommendations():
    recommendations = _result()["recommendations"]
    for item in recommendations:
        if item["timestamp"] is not None:
            item["timestamp"] = str(item["timestamp"])
    return recommendations


@app.get("/get-locations")
def get_locations():
    return _result()["locations"].to_dict(orient="records")


@app.get("/demo-scenario")
def demo_scenario():
    result = _result()
    hourly = result["hourly"].copy()
    peak = hourly.sort_values("risk_ratio", ascending=False).iloc[0]
    locations = result["locations"].head(5)

    return {
        "current_demand": round(float(hourly["predicted_demand_kwh"].mean()), 2),
        "highest_risk_time": str(peak["timestamp"]),
        "before_peak_kwh": round(float(peak["predicted_demand_kwh"]), 2),
        "after_peak_kwh": round(float(peak["after_shift_kwh"]), 2),
        "peak_reduction_kwh": round(float(peak["shiftable_kwh"]), 2),
        "recommendations": result["recommendations"],
        "top_station_zones": locations["location_id"].tolist(),
    }

