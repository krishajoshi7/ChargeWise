from __future__ import annotations

from chargewise.data import ensure_dataset, make_future_frame
from chargewise.model import load_model, predict_demand, train_model
from chargewise.recommendations import build_schedule_recommendations, detect_peak_hours
from chargewise.station_placement import rank_station_locations


def run_pipeline(train: bool = True) -> dict:
    df = ensure_dataset()
    metrics = train_model(df) if train else {"status": "skipped"}
    model = load_model()
    future = make_future_frame()
    predictions = predict_demand(model, future)
    hourly = detect_peak_hours(predictions)
    recommendations = build_schedule_recommendations(hourly)
    locations = rank_station_locations(predictions)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "hourly": hourly,
        "recommendations": recommendations,
        "locations": locations,
    }


def main() -> None:
    result = run_pipeline(train=True)
    print("ChargeWise pipeline complete")
    print(f"Rows trained: {result['metrics']['rows']}")
    print(f"MAE: {result['metrics']['mae']} kWh")
    print(f"R2: {result['metrics']['r2']}")
    print("Top station locations:")
    print(result["locations"][["location_id", "priority_score"]].to_string(index=False))


if __name__ == "__main__":
    main()

