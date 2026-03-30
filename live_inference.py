from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from standalone_inference import DEFAULT_LOOKBACK_MINUTES, generate_prediction_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live NEM spot-price inference.")
    parser.add_argument(
        "--models-dir",
        default="models/multi_horizon",
        help="Directory containing residual-blend model pickle and metadata files.",
    )
    parser.add_argument(
        "--log-path",
        default="predictions/prediction_log.csv",
        help="CSV file to append prediction rows to.",
    )
    parser.add_argument(
        "--latest-output",
        default="predictions/latest_predictions.json",
        help="JSON file containing the latest prediction payload.",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=DEFAULT_LOOKBACK_MINUTES,
        help="Minutes of recent NEM data to export before feature generation.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENELECTRICITY_API_KEY"),
        help="OpenElectricity API key. Defaults to OPENELECTRICITY_API_KEY if set.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENELECTRICITY_API_URL", "https://api.openelectricity.org.au/v4"),
        help="OpenElectricity API base URL.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing OpenElectricity API key. Set OPENELECTRICITY_API_KEY or pass --api-key.")

    folder_root = Path(__file__).resolve().parent
    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = folder_root / models_dir
    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = folder_root / log_path
    latest_output = Path(args.latest_output)
    if not latest_output.is_absolute():
        latest_output = folder_root / latest_output

    latest_payload = generate_prediction_payload(
        api_key=args.api_key,
        models_dir=models_dir,
        latest_output=latest_output,
        log_path=log_path,
        lookback_minutes=args.lookback_minutes,
        base_url=args.base_url,
    )

    print(json.dumps(latest_payload, indent=2))


if __name__ == "__main__":
    main()
