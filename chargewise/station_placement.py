from __future__ import annotations

import pandas as pd


def rank_station_locations(predictions: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    ranked = (
        predictions.groupby(["location_id", "lat", "lon"], as_index=False)
        .agg(
            total_predicted_kwh=("predicted_demand_kwh", "sum"),
            max_hourly_kwh=("predicted_demand_kwh", "max"),
            avg_ev_count=("ev_count", "mean"),
        )
        .sort_values("total_predicted_kwh", ascending=False)
    )
    ranked["priority_score"] = (
        ranked["total_predicted_kwh"] * 0.65
        + ranked["max_hourly_kwh"] * 2.5
        + ranked["avg_ev_count"] * 8
    ).round(2)
    ranked = ranked.sort_values("priority_score", ascending=False).head(top_n)
    ranked["recommendation"] = ranked["location_id"].apply(
        lambda zone: f"Install or expand charging capacity near {zone.replace('_', ' ')}."
    )
    return ranked.reset_index(drop=True)

