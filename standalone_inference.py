from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
MARKET_TIME_OFFSET_HOURS = 10
HORIZONS: dict[int, str] = {
    1: "5m",
    3: "15m",
    6: "30m",
}
LOAD_FUELTECHS = {"pumps", "battery_charging"}
RENEWABLE_FUELTECHS = {
    "solar_utility",
    "solar_rooftop",
    "wind",
    "hydro",
    "battery_discharging",
    "bioenergy_biogas",
    "bioenergy_biomass",
}
DEFAULT_BASE_URL = os.environ.get("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4")
DEFAULT_LOOKBACK_MINUTES = 90
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models" / "multi_horizon"
DEFAULT_LATEST_OUTPUT = Path(__file__).resolve().parent / "predictions" / "latest_predictions.json"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "predictions" / "prediction_log.csv"


def strip_timezone(dt: datetime) -> str:
    return dt.replace(tzinfo=None, microsecond=0).isoformat()


def to_float(value: Any) -> float:
    return float("nan") if value is None else float(value)


def mean(values: list[float | None]) -> float:
    filtered = [value for value in values if value is not None and not np.isnan(value)]
    if not filtered:
        return float("nan")
    return float(sum(filtered) / len(filtered))


def sample_std(values: list[float | None]) -> float:
    filtered = np.array([value for value in values if value is not None and not np.isnan(value)], dtype=float)
    if filtered.size < 2:
        return float("nan")
    return float(filtered.std(ddof=1))


def api_request(path: str, params: list[tuple[str, str]], api_key: str, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    response = requests.get(
        f"{base_url}{path}",
        params=params,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "nem-spot-price-predictor/1.0",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def network_time_series_to_rows(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_map: dict[tuple[str, tuple[tuple[str, Any], ...]], dict[str, Any]] = {}

    for series in data:
        metric = series["metric"]
        for result in series.get("results", []):
            columns = result.get("columns", {})
            grouping_items = tuple(sorted(columns.items()))
            for timestamp, value in result.get("data", []):
                row_key = (timestamp, grouping_items)
                row = rows_map.setdefault(
                    row_key,
                    {
                        "interval": timestamp,
                        **columns,
                    },
                )
                row[metric] = value

    return sorted(rows_map.values(), key=lambda row: row["interval"])


def get_market_time_parts(timestamp: str) -> dict[str, int]:
    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    market_ts = ts + timedelta(hours=MARKET_TIME_OFFSET_HOURS)
    js_day_of_week = (market_ts.weekday() + 1) % 7
    is_weekend = 1 if js_day_of_week in {0, 6} else 0
    is_business_hour = 1 if is_weekend == 0 and 9 <= market_ts.hour < 17 else 0
    return {
        "hour": market_ts.hour,
        "day_of_week": js_day_of_week,
        "month": market_ts.month,
        "is_weekend": is_weekend,
        "is_business_hour": is_business_hour,
    }


def fetch_recent_region_series(api_key: str, lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES, base_url: str = DEFAULT_BASE_URL) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    date_start = strip_timezone(now - timedelta(minutes=lookback_minutes))

    generation_response = api_request(
        "/data/network/NEM",
        [
            ("metrics", "power"),
            ("metrics", "energy"),
            ("metrics", "market_value"),
            ("interval", "5m"),
            ("date_start", date_start),
            ("primary_grouping", "network_region"),
            ("secondary_grouping", "fueltech"),
        ],
        api_key,
        base_url,
    )
    market_response = api_request(
        "/market/network/NEM",
        [
            ("metrics", "price"),
            ("metrics", "demand"),
            ("metrics", "curtailment_solar_utility"),
            ("metrics", "curtailment_wind"),
            ("interval", "5m"),
            ("date_start", date_start),
            ("primary_grouping", "network_region"),
        ],
        api_key,
        base_url,
    )

    generation_rows = network_time_series_to_rows(generation_response.get("data", []))
    market_rows = network_time_series_to_rows(market_response.get("data", []))

    generation_by_ts_region: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        region = row.get("network_region") or row.get("region")
        if region in REGIONS:
            generation_by_ts_region[(row["interval"], region)].append(row)

    market_by_ts_region: dict[tuple[str, str], dict[str, Any]] = {}
    for row in market_rows:
        region = row.get("network_region") or row.get("region")
        if region in REGIONS:
            market_by_ts_region[(row["interval"], region)] = row

    timestamps = sorted({row["interval"] for row in generation_rows} | {row["interval"] for row in market_rows})
    region_snapshots: list[dict[str, Any]] = []

    for timestamp in timestamps:
        row: dict[str, Any] = {"timestamp": timestamp}
        for region in REGIONS:
            generation_items = generation_by_ts_region.get((timestamp, region), [])
            market_item = market_by_ts_region.get((timestamp, region), {})
            current_price = to_float(market_item.get("price"))
            current_demand = to_float(market_item.get("demand"))
            wind_mw = 0.0
            total_net_power = 0.0
            renewables_mw = 0.0

            for item in generation_items:
                fueltech = str(item.get("fueltech", ""))
                power = to_float(item.get("power"))
                if fueltech in LOAD_FUELTECHS:
                    continue
                total_net_power += max(0.0, power)
                if fueltech in RENEWABLE_FUELTECHS:
                    renewables_mw += power
                if fueltech == "wind":
                    wind_mw = power

            renewables_pct = (renewables_mw / total_net_power * 100.0) if total_net_power > 0 else float("nan")

            row[f"{region}_price_dollar_per_mwh"] = current_price
            row[f"{region}_demand_mw"] = current_demand
            row[f"{region}_renewables_pct"] = renewables_pct
            row[f"{region}_gen_wind_mw"] = wind_mw

        region_snapshots.append(row)

    return region_snapshots


def build_feature_row(region_series: list[dict[str, Any]]) -> dict[str, float | int | str]:
    if len(region_series) < 13:
        raise ValueError(f"Need at least 13 recent 5-minute intervals, got {len(region_series)}.")

    snapshots = region_series[-13:]
    feature_row: dict[str, float | int | str] = {"timestamp": snapshots[-1]["timestamp"]}
    feature_row.update(get_market_time_parts(str(snapshots[-1]["timestamp"])))

    for region in REGIONS:
        prices = [to_float(snapshot.get(f"{region}_price_dollar_per_mwh")) for snapshot in snapshots]
        demands = [to_float(snapshot.get(f"{region}_demand_mw")) for snapshot in snapshots]
        renewables = [to_float(snapshot.get(f"{region}_renewables_pct")) for snapshot in snapshots]
        winds = [to_float(snapshot.get(f"{region}_gen_wind_mw")) for snapshot in snapshots]

        current_price = prices[-1]
        current_demand = demands[-1]
        current_renewables = renewables[-1]
        current_wind = winds[-1]

        feature_row[f"{region}_price_dollar_per_mwh"] = current_price
        feature_row[f"{region}_demand_mw"] = current_demand
        feature_row[f"{region}_renewables_pct"] = current_renewables
        feature_row[f"{region}_price_lag_1"] = prices[-2]
        feature_row[f"{region}_price_lag_2"] = prices[-3]
        feature_row[f"{region}_price_lag_6"] = prices[-7]
        feature_row[f"{region}_price_lag_12"] = prices[-13]
        feature_row[f"{region}_demand_rollmean_6"] = mean(demands[-6:])
        feature_row[f"{region}_demand_rollmean_12"] = mean(demands[-12:])
        feature_row[f"{region}_wind_rollmean_6"] = mean(winds[-6:])
        feature_row[f"{region}_wind_rollmean_12"] = mean(winds[-12:])
        feature_row[f"{region}_gen_wind_mw"] = current_wind

        feature_row[f"{region}_price_ramp_1"] = current_price - prices[-2]
        feature_row[f"{region}_price_ramp_2"] = current_price - prices[-3]
        feature_row[f"{region}_price_ramp_6"] = current_price - prices[-7]
        feature_row[f"{region}_price_ramp_12"] = current_price - prices[-13]
        feature_row[f"{region}_price_ramp_1_to_6"] = prices[-2] - prices[-7]
        feature_row[f"{region}_price_volatility"] = sample_std([prices[-1], prices[-2], prices[-3], prices[-7], prices[-13]])
        feature_row[f"{region}_demand_ramp_6"] = current_demand - feature_row[f"{region}_demand_rollmean_6"]
        feature_row[f"{region}_demand_ramp_12"] = current_demand - feature_row[f"{region}_demand_rollmean_12"]
        feature_row[f"{region}_wind_ramp_6"] = current_wind - feature_row[f"{region}_wind_rollmean_6"]
        feature_row[f"{region}_wind_ramp_12"] = current_wind - feature_row[f"{region}_wind_rollmean_12"]
        feature_row[f"{region}_wind_share_change"] = current_wind * current_renewables / 100.0

    feature_row["hour_sin"] = float(np.sin(2 * np.pi * float(feature_row["hour"]) / 24))
    feature_row["hour_cos"] = float(np.cos(2 * np.pi * float(feature_row["hour"]) / 24))
    feature_row["dow_sin"] = float(np.sin(2 * np.pi * float(feature_row["day_of_week"]) / 7))
    feature_row["dow_cos"] = float(np.cos(2 * np.pi * float(feature_row["day_of_week"]) / 7))

    for left, right in combinations(REGIONS, 2):
        left_price = float(feature_row[f"{left}_price_dollar_per_mwh"])
        right_price = float(feature_row[f"{right}_price_dollar_per_mwh"])
        spread = left_price - right_price
        feature_row[f"spread_{left}_{right}"] = spread
        feature_row[f"abs_spread_{left}_{right}"] = abs(spread)

    return feature_row


def load_metadata(models_dir: Path, region: str, horizon_steps: int) -> dict[str, Any]:
    metadata_path = models_dir / f"{region.lower()}_lightgbm_residual_blend_tplus{horizon_steps}_metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def predict_residual(model: Any, model_input: pd.DataFrame) -> float:
    booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
    if booster is not None:
        return float(booster.predict(model_input)[0])
    return float(model.predict(model_input)[0])


def append_prediction_log(log_path: Path, rows: list[dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prediction_generated_at",
        "source_snapshot_at",
        "region",
        "horizon_label",
        "horizon_minutes",
        "model_name",
        "alpha",
        "current_price",
        "predicted_residual",
        "predicted_price",
    ]
    expected_header = ",".join(fieldnames)
    file_mode = "a"
    write_header = not log_path.exists()

    if log_path.exists():
        with log_path.open("r", encoding="utf-8", newline="") as handle:
            first_line = handle.readline().strip()
        if first_line != expected_header:
            file_mode = "w"
            write_header = True

    with log_path.open(file_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def generate_prediction_payload(
    api_key: str,
    models_dir: Path = DEFAULT_MODELS_DIR,
    latest_output: Path | None = DEFAULT_LATEST_OUTPUT,
    log_path: Path | None = DEFAULT_LOG_PATH,
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    region_series = fetch_recent_region_series(api_key, lookback_minutes=lookback_minutes, base_url=base_url)
    feature_row = build_feature_row(region_series)
    source_snapshot_at = str(feature_row["timestamp"])
    prediction_generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    region_predictions: list[dict[str, Any]] = []
    prediction_log_rows: list[dict[str, Any]] = []

    for region in REGIONS:
        current_price = float(feature_row[f"{region}_price_dollar_per_mwh"])
        forecasts: dict[str, Any] = {}

        for horizon_steps, horizon_label in HORIZONS.items():
            metadata = load_metadata(models_dir, region, horizon_steps)
            model_path = models_dir / f"{region.lower()}_lightgbm_residual_blend_tplus{horizon_steps}.pkl"
            model = joblib.load(model_path)
            feature_columns = metadata["feature_columns"]
            alpha = float(metadata["alpha"])
            model_input = pd.DataFrame([{column: feature_row.get(column, np.nan) for column in feature_columns}])
            predicted_residual = predict_residual(model, model_input)
            predicted_price = current_price + alpha * predicted_residual

            forecasts[horizon_label] = {
                "model_name": metadata["model_name"],
                "alpha": alpha,
                "horizon_steps": horizon_steps,
                "horizon_minutes": int(metadata.get("horizon_minutes", horizon_steps * 5)),
                "predicted_residual": predicted_residual,
                "predicted_price": predicted_price,
            }

            prediction_log_rows.append(
                {
                    "prediction_generated_at": prediction_generated_at,
                    "source_snapshot_at": source_snapshot_at,
                    "region": region,
                    "horizon_label": horizon_label,
                    "horizon_minutes": forecasts[horizon_label]["horizon_minutes"],
                    "model_name": metadata["model_name"],
                    "alpha": alpha,
                    "current_price": current_price,
                    "predicted_residual": predicted_residual,
                    "predicted_price": predicted_price,
                }
            )

        region_predictions.append(
            {
                "prediction_generated_at": prediction_generated_at,
                "source_snapshot_at": source_snapshot_at,
                "region": region,
                "current_price": current_price,
                "forecasts": forecasts,
            }
        )

    latest_payload = {
        "prediction_generated_at": prediction_generated_at,
        "source_snapshot_at": source_snapshot_at,
        "prediction_horizons_minutes": [steps * 5 for steps in HORIZONS.keys()],
        "regions": region_predictions,
    }

    if log_path is not None:
        append_prediction_log(log_path, prediction_log_rows)
    if latest_output is not None:
        latest_output.parent.mkdir(parents=True, exist_ok=True)
        latest_output.write_text(f"{json.dumps(latest_payload, indent=2)}\n", encoding="utf-8")

    return latest_payload
