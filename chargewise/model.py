from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from chargewise.config import MODEL_PATH

FEATURES = [
    "location_id",
    "hour",
    "day_of_week",
    "day_type",
    "ev_count",
    "weather_temp_c",
    "rain_mm",
]
TARGET = "demand_kwh"


def train_model(df: pd.DataFrame, model_path=MODEL_PATH) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["location_id", "day_type"]),
            ("numeric", "passthrough", ["hour", "day_of_week", "ev_count", "weather_temp_c", "rain_mm"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=180,
                    random_state=42,
                    min_samples_leaf=3,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "r2": round(float(r2_score(y_test, predictions)), 3),
        "rows": int(len(df)),
    }


def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python -m chargewise.pipeline` first.")
    return joblib.load(model_path)


def predict_demand(model, future_df: pd.DataFrame) -> pd.DataFrame:
    result = future_df.copy()
    predicted = model.predict(result[FEATURES])
    result["predicted_demand_kwh"] = predicted.round(2)
    result["lower_bound_kwh"] = (result["predicted_demand_kwh"] * 0.9).round(2)
    result["upper_bound_kwh"] = (result["predicted_demand_kwh"] * 1.1).round(2)
    return result
