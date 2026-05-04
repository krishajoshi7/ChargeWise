from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from chargewise.config import DATA_PATH, ZONES


def _hour_pattern(hour: int) -> float:
    morning = 0.55 * np.exp(-((hour - 8) ** 2) / 10)
    evening = 1.25 * np.exp(-((hour - 19) ** 2) / 8)
    late_night = 0.35 * np.exp(-((hour - 23) ** 2) / 16)
    return 0.45 + morning + evening + late_night


def generate_synthetic_data(
    start: str = "2026-01-01",
    periods: int = 24 * 90,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=periods, freq="h")
    rows = []

    for ts in timestamps:
        is_weekend = ts.dayofweek >= 5
        day_multiplier = 0.9 if is_weekend else 1.0
        weather_temp_c = 21 + 7 * np.sin((ts.hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 1.6)
        rain_mm = max(0, rng.normal(0.5 if is_weekend else 0.2, 1.2))

        for zone in ZONES:
            base_ev_count = 32 * zone["density"] * day_multiplier
            commute_boost = 1 + 0.25 * np.exp(-((ts.hour - 19) ** 2) / 10)
            ev_count = max(6, int(rng.normal(base_ev_count * commute_boost, 5)))
            demand = ev_count * 3.2 * _hour_pattern(ts.hour) * zone["density"]
            demand += 0.9 * max(weather_temp_c - 28, 0) * zone["density"]
            demand += rain_mm * 1.8
            demand += rng.normal(0, 10)
            demand_kwh = max(12, round(demand, 2))

            rows.append(
                {
                    "timestamp": ts,
                    "location_id": zone["location_id"],
                    "lat": zone["lat"],
                    "lon": zone["lon"],
                    "hour": ts.hour,
                    "day_of_week": ts.dayofweek,
                    "day_type": "weekend" if is_weekend else "weekday",
                    "ev_count": ev_count,
                    "weather_temp_c": round(weather_temp_c, 1),
                    "rain_mm": round(rain_mm, 2),
                    "demand_kwh": demand_kwh,
                }
            )

    return pd.DataFrame(rows)


def ensure_dataset(path=DATA_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])

    df = generate_synthetic_data()
    df.to_csv(path, index=False)
    return df


def make_future_frame(hours: int = 24, start: datetime | None = None) -> pd.DataFrame:
    if start is None:
        now = pd.Timestamp.now().floor("h")
        start = now + pd.Timedelta(hours=1)

    timestamps = pd.date_range(start=start, periods=hours, freq="h")
    rows = []
    for ts in timestamps:
        is_weekend = ts.dayofweek >= 5
        for zone in ZONES:
            density = zone["density"]
            hour_factor = _hour_pattern(ts.hour)
            ev_count = int(max(8, 32 * density * (0.92 if is_weekend else 1.0) * (0.85 + hour_factor / 3)))
            weather_temp_c = 24 + 5 * np.sin((ts.hour - 6) / 24 * 2 * np.pi)
            rows.append(
                {
                    "timestamp": ts,
                    "location_id": zone["location_id"],
                    "lat": zone["lat"],
                    "lon": zone["lon"],
                    "hour": ts.hour,
                    "day_of_week": ts.dayofweek,
                    "day_type": "weekend" if is_weekend else "weekday",
                    "ev_count": ev_count,
                    "weather_temp_c": round(weather_temp_c, 1),
                    "rain_mm": 0.2 if not is_weekend else 0.4,
                }
            )
    return pd.DataFrame(rows)

