from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import requests
from standalone_inference import DEFAULT_MODELS_DIR, generate_prediction_payload

LOCAL_PREDICTIONS_PATH = Path(__file__).resolve().parent / "predictions" / "latest_predictions.json"
PREDICTION_API_URL = os.environ.get("PREDICTION_API_URL", "").strip()
PREDICTION_API_KEY = os.environ.get("PREDICTION_API_KEY", "").strip()
OPENELECTRICITY_API_KEY = os.environ.get("OPENELECTRICITY_API_KEY", "").strip()
OPENELECTRICITY_API_URL = os.environ.get("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4").strip()


def load_prediction_payload() -> dict[str, Any]:
    if PREDICTION_API_URL:
        headers = {"x-api-key": PREDICTION_API_KEY} if PREDICTION_API_KEY else {}
        response = requests.get(PREDICTION_API_URL, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload["events"][0]["attribute"] if "events" in payload else payload

    if OPENELECTRICITY_API_KEY:
        return generate_prediction_payload(
            api_key=OPENELECTRICITY_API_KEY,
            models_dir=DEFAULT_MODELS_DIR,
            latest_output=LOCAL_PREDICTIONS_PATH,
            log_path=Path(__file__).resolve().parent / "predictions" / "prediction_log.csv",
            base_url=OPENELECTRICITY_API_URL,
        )

    if not LOCAL_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "No prediction source found. Set OPENELECTRICITY_API_KEY for live inference, "
            f"set PREDICTION_API_URL, or create {LOCAL_PREDICTIONS_PATH}."
        )

    return json.loads(LOCAL_PREDICTIONS_PATH.read_text(encoding="utf-8"))




def build_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    
    # Clean naming for our dashboard columns
    TARGET_METRIC_LABELS = {
        "price": "Price",
        "demand": "Demand",
        "gen_wind": "Wind Gen",
        "gen_coal_black": "Black Coal",
        "gen_coal_brown": "Brown Coal",
        "gen_solar_utility": "Solar (Grid)",
        "gen_solar_rooftop": "Solar (Rooftop)",
        "gen_hydro": "Hydro",
        "gen_battery_discharging": "Battery",
        "renewables_pct": "Renewables %",
    }

    for row in payload.get("regions", []):
        forecasts = row.get("forecasts", {})
        current_values = row.get("current_values", {})
        
        flat_row = {
            "Region": row["region"],
            "Prediction Generated At": row.get("prediction_generated_at", payload.get("prediction_generated_at")),
            "Source Snapshot At": row.get("source_snapshot_at", payload.get("source_snapshot_at")),
        }
        
        for metric, label in TARGET_METRIC_LABELS.items():
            if metric not in current_values:
                continue
                
            flat_row[f"Current {label}"] = current_values.get(metric)
            flat_row[f"{label} In 5m"] = forecasts.get("5m", {}).get(metric, {}).get("predicted_value")
            flat_row[f"{label} In 15m"] = forecasts.get("15m", {}).get(metric, {}).get("predicted_value")
            flat_row[f"{label} In 30m"] = forecasts.get("30m", {}).get(metric, {}).get("predicted_value")

        rows.append(flat_row)

    return pd.DataFrame(rows)


def refresh_dashboard() -> tuple[str, pd.DataFrame, dict[str, Any]]:
    payload = load_prediction_payload()
    table = build_dataframe(payload)
    summary = (
        f"Prediction generated at: {payload.get('prediction_generated_at')} | "
        f"Source snapshot at: {payload.get('source_snapshot_at')} | "
        f"Horizons: {payload.get('prediction_horizons_minutes')}"
    )
    return summary, table, payload


with gr.Blocks(title="NEM Spot Price Predictor") as demo:
    gr.Markdown("# NEM Spot Price Predictor")
    gr.Markdown(
         "Shows current regional generation baselines and model-predicted targets (Price, Demand, Wind, Coal) in 5, 15, and 30 minutes."
    )

    summary = gr.Textbox(label="Summary", interactive=False)
    prediction_table = gr.Dataframe(label="Regional Predictions", interactive=False)
    raw_payload = gr.JSON(label="Raw Prediction Payload")

    refresh_button = gr.Button("Refresh Predictions")
    refresh_button.click(refresh_dashboard, outputs=[summary, prediction_table, raw_payload])

    demo.load(refresh_dashboard, outputs=[summary, prediction_table, raw_payload])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
