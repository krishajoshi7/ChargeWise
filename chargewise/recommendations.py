from __future__ import annotations

import pandas as pd

from chargewise.config import GRID_CAPACITY_KWH, PEAK_THRESHOLD_RATIO, SHIFT_RATIO


def detect_peak_hours(predictions: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        predictions.groupby("timestamp", as_index=False)["predicted_demand_kwh"]
        .sum()
        .sort_values("timestamp")
    )
    hourly["capacity_kwh"] = GRID_CAPACITY_KWH * predictions["location_id"].nunique()
    hourly["risk_ratio"] = hourly["predicted_demand_kwh"] / hourly["capacity_kwh"]
    hourly["is_peak"] = hourly["risk_ratio"] >= PEAK_THRESHOLD_RATIO
    hourly["shiftable_kwh"] = 0.0
    hourly.loc[hourly["is_peak"], "shiftable_kwh"] = (
        hourly.loc[hourly["is_peak"], "predicted_demand_kwh"] * SHIFT_RATIO
    ).round(2)
    hourly["after_shift_kwh"] = (
        hourly["predicted_demand_kwh"] - hourly["shiftable_kwh"]
    ).round(2)
    return hourly


def build_schedule_recommendations(hourly: pd.DataFrame) -> list[dict]:
    peak_rows = hourly[hourly["is_peak"]].copy()
    recommendations = []

    for _, row in peak_rows.iterrows():
        hour_label = pd.Timestamp(row["timestamp"]).strftime("%I %p").lstrip("0")
        recommendations.append(
            {
                "timestamp": row["timestamp"],
                "risk": round(float(row["risk_ratio"]), 2),
                "message": (
                    f"Grid risk detected around {hour_label}. "
                    f"Shift about {row['shiftable_kwh']:.0f} kWh to 10 PM-6 AM."
                ),
                "explanation": "High demand due to evening commute and home charging pattern.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "timestamp": None,
                "risk": 0,
                "message": "No critical peak detected in the next 24 hours.",
                "explanation": "Demand remains below the configured grid-risk threshold.",
            }
        )

    return recommendations

