#!/usr/bin/env python
"""
This script trains a Random Forest model and exports it using MLflow.
"""
import argparse
import logging
import os
import shutil

import mlflow
import json

import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
import sklearn.pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def delta_date_feature(dates):
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(
        lambda d: (d.max() - d).dt.days, axis=0
    ).to_numpy()


def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    rf_config['random_state'] = args.random_seed

    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    logger.info("Fitting the pipeline")
    sk_pipe.fit(X_train, y_train)

    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    signature = infer_signature(X_val, y_pred)
    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        signature=signature,
        input_example=X_val.iloc[:5]
    )

    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Random Forest pipeline export",
    )
    artifact.add_dir("random_forest_dir")
    run.log_artifact(artifact)
    artifact.wait()

    run.summary['r2'] = r_squared
    run.summary['mae'] = mae


def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]

    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = sklearn.pipeline.make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    date_imputer = sklearn.pipeline.make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = sklearn.pipeline.make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            sublinear_tf=True,
        ),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ["ordinal_cat", ordinal_categorical_preproc, ordinal_categorical],
            ["non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical],
            ["impute_zero", zero_imputer, zero_imputed],
            ["transform_date", date_imputer, ["last_review"]],
            ["transform_name", name_tfidf, ["name"]],
        ],
        remainder="drop",
    )

    processed_features = (
        ordinal_categorical
        + non_ordinal_categorical
        + zero_imputed
        + ["last_review", "name"]
    )

    sk_pipe = sklearn.pipeline.Pipeline(
        steps=[
            ["preprocessor", preprocessor],
            ["random_forest", RandomForestRegressor(**rf_config)],
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a Random Forest model")

    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, default=42, required=False)
    parser.add_argument("--stratify_by", type=str, default="none", required=False)
    parser.add_argument("--rf_config", type=str, default="{}")
    parser.add_argument("--max_tfidf_features", type=int, default=10)
    parser.add_argument("--output_artifact", type=str, required=True)

    args = parser.parse_args()

    go(args)
