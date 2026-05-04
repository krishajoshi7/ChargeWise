from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chargewise.pipeline import run_pipeline

st.set_page_config(page_title="ChargeWise", layout="wide")


@st.cache_data(ttl=300)
def load_result():
    return run_pipeline(train=False)


try:
    result = load_result()
except FileNotFoundError:
    st.warning("Model not found. Training the MVP model now.")
    result = run_pipeline(train=True)

predictions = result["predictions"]
hourly = result["hourly"]
locations = result["locations"]
recommendations = result["recommendations"]

st.title("ChargeWise")
st.caption("EV charging demand prediction, peak reduction, and station placement support")

peak_count = int(hourly["is_peak"].sum())
max_peak = hourly.sort_values("predicted_demand_kwh", ascending=False).iloc[0]
top_zone = locations.iloc[0]

metric_cols = st.columns(4)
metric_cols[0].metric("Next 24h demand", f"{hourly['predicted_demand_kwh'].sum():,.0f} kWh")
metric_cols[1].metric("Peak-risk hours", peak_count)
metric_cols[2].metric("Max hourly demand", f"{max_peak['predicted_demand_kwh']:,.0f} kWh")
metric_cols[3].metric("Top station zone", top_zone["location_id"].replace("_", " "))

left, right = st.columns([1.55, 1])

with left:
    st.subheader("Before vs After Smart Charging")
    chart_df = hourly.melt(
        id_vars=["timestamp", "is_peak"],
        value_vars=["predicted_demand_kwh", "after_shift_kwh"],
        var_name="scenario",
        value_name="demand_kwh",
    )
    chart_df["scenario"] = chart_df["scenario"].map(
        {
            "predicted_demand_kwh": "Before scheduling",
            "after_shift_kwh": "After recommendation",
        }
    )
    fig = px.line(
        chart_df,
        x="timestamp",
        y="demand_kwh",
        color="scenario",
        markers=True,
        labels={"timestamp": "Time", "demand_kwh": "Demand (kWh)", "scenario": ""},
    )
    fig.add_hrect(
        y0=float(hourly["capacity_kwh"].iloc[0] * 0.8),
        y1=float(hourly["capacity_kwh"].iloc[0] * 1.2),
        fillcolor="rgba(220, 80, 55, 0.12)",
        line_width=0,
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Decision Panel")
    for item in recommendations[:5]:
        st.info(item["message"])
        st.caption(item["explanation"])

    st.subheader("Demo Scenario")
    st.write(
        f"Peak demand reaches **{max_peak['predicted_demand_kwh']:,.0f} kWh**. "
        f"ChargeWise shifts **{max_peak['shiftable_kwh']:,.0f} kWh** to off-peak hours, "
        f"reducing the peak to **{max_peak['after_shift_kwh']:,.0f} kWh**."
    )

st.subheader("Demand Heatmap and Recommended Station Locations")
map_points = (
    predictions.groupby(["location_id", "lat", "lon"], as_index=False)
    .agg(predicted_demand_kwh=("predicted_demand_kwh", "sum"))
)
map_points["radius"] = (map_points["predicted_demand_kwh"] / map_points["predicted_demand_kwh"].max()) * 900
station_points = locations.copy()

deck = pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    initial_view_state=pdk.ViewState(latitude=12.97, longitude=77.62, zoom=10.2, pitch=35),
    layers=[
        pdk.Layer(
            "HeatmapLayer",
            data=map_points,
            get_position="[lon, lat]",
            get_weight="predicted_demand_kwh",
            radius_pixels=65,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=station_points,
            get_position="[lon, lat]",
            get_radius=220,
            get_fill_color="[15, 118, 110, 210]",
            pickable=True,
        ),
    ],
    tooltip={"text": "{location_id}\nDemand: {total_predicted_kwh} kWh"},
)
st.pydeck_chart(deck, use_container_width=True)

table_cols = [
    "location_id",
    "total_predicted_kwh",
    "max_hourly_kwh",
    "priority_score",
    "recommendation",
]
st.subheader("Top 5 New Station Recommendations")
st.dataframe(locations[table_cols], use_container_width=True, hide_index=True)

with st.expander("Model Details"):
    st.write(result["metrics"])
    hourly_display = hourly.copy()
    hourly_display["timestamp"] = hourly_display["timestamp"].astype(str)
    st.dataframe(hourly_display, use_container_width=True, hide_index=True)
