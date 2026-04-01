# Electricity Grid Pricing Prediction Model

Machine learning model for forecasting Australian electricity spot prices.

## Overview

This repository contains the implementation of a pricing prediction model deployed on Hugging Face. It generates short-term forecasts for electricity prices based on historical and real-time data.

## Features

- Predicts spot prices at:
  - 5 minutes
  - 15 minutes
  - 30 minutes
- Integrated with external APIs for live inference
- Deployable via Hugging Face Spaces

## Model Deployment

The model is hosted on Hugging Face and exposed via an API endpoint for inference.

## Input

- Time-series electricity data
- Market indicators

## Output

- Predicted spot prices for defined time horizons

## Integration

This model integrates with:
- Electricity Grid API (data input)
- AWS Lambda (automated testing and triggering)

## Usage

The model can be accessed via API calls or integrated into downstream applications such as dashboards or trading tools.

## Notes

- Ensure input data is preprocessed consistently
- Model performance depends on data freshness
