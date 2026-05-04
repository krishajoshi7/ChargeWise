from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "synthetic_ev_demand.csv"
MODEL_PATH = MODEL_DIR / "demand_model.joblib"

CITY_CENTER = (12.9716, 77.5946)
GRID_CAPACITY_KWH = 260.0
PEAK_THRESHOLD_RATIO = 0.8
SHIFT_RATIO = 0.3

ZONES = [
    {"location_id": "Zone_A_Indiranagar", "lat": 12.9784, "lon": 77.6408, "density": 1.22},
    {"location_id": "Zone_B_Whitefield", "lat": 12.9698, "lon": 77.7500, "density": 1.35},
    {"location_id": "Zone_C_ElectronicCity", "lat": 12.8452, "lon": 77.6602, "density": 1.18},
    {"location_id": "Zone_D_MG_Road", "lat": 12.9756, "lon": 77.6068, "density": 1.28},
    {"location_id": "Zone_E_Yelahanka", "lat": 13.1007, "lon": 77.5963, "density": 0.82},
    {"location_id": "Zone_F_Jayanagar", "lat": 12.9250, "lon": 77.5938, "density": 0.98},
    {"location_id": "Zone_G_Hebbal", "lat": 13.0358, "lon": 77.5970, "density": 1.05},
    {"location_id": "Zone_H_Koramangala", "lat": 12.9352, "lon": 77.6245, "density": 1.25},
]

