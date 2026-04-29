# Build a Machine Learning Pipeline for Short-Term Rental Prices in NYC

W&B Project: https://wandb.ai/jason-zeman7-/nyc_airbnb
GitHub: https://github.com/jasonzeman7/Project-Build-an-ML-Pipeline-Starter

## Project Overview

This project builds an end-to-end machine learning pipeline to estimate typical rental prices for short-term properties in NYC. The pipeline is designed to be rerun weekly as new data comes in.

## Pipeline Steps

1. Download - Fetches raw data and uploads to W&B
2. Basic Cleaning - Removes price outliers and properties outside NYC boundaries
3. Data Check - Runs tests to verify data quality
4. Data Split - Splits data into train/validation and test sets
5. Train Random Forest - Trains and exports the model
6. Test Regression Model - Verifies model performance on the test set

## Running the Pipeline

Run the full pipeline:
mlflow run .

Run a specific step:
mlflow run . -P steps=download

## Model Performance

Validation MAE: 37.35
Test MAE: 36.86
No overfitting observed
