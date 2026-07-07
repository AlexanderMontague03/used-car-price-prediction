"""Prediction helpers for the used-car price dashboard."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "catboost_used_car_price_quantile_model.pkl"

NUMERIC_FEATURES = [
    "year",
    "kilometres",
    "engine_size_litres",
    "fuel_consumption_per_100km",
    "cylinders",
    "doors",
    "seats",
]

CATEGORICAL_FEATURES = [
    "brand",
    "transmission",
    "body_type",
    "fuel_type",
    "drive_type",
    "state",
    "colour_ext",
    "model",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

INTERVAL_COLUMNS = {
    "P10-P90": ("price_p10", "price_p90"),
    "P05-P95": ("price_p05", "price_p95"),
}


@lru_cache(maxsize=1)
def load_price_model(model_path=DEFAULT_MODEL_PATH):
    """Load the saved CatBoost MultiQuantile model bundle."""
    return joblib.load(model_path)


def _prepare_prediction_input(user_input):
    """Convert a dashboard input dictionary into the model's expected DataFrame."""
    input_row = {feature: user_input.get(feature, np.nan) for feature in MODEL_FEATURES}
    input_df = pd.DataFrame([input_row], columns=MODEL_FEATURES)

    for col in NUMERIC_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_df[CATEGORICAL_FEATURES] = (
        input_df[CATEGORICAL_FEATURES]
        .astype("object")
        .where(pd.notna(input_df[CATEGORICAL_FEATURES]), np.nan)
    )

    return input_df


def _predict_quantiles(model_bundle, input_df):
    """Return a one-row DataFrame containing all quantile predictions."""
    feature_names = model_bundle.get("feature_names", MODEL_FEATURES)
    quantiles = model_bundle["quantiles"]

    X_prepared = model_bundle["preprocessor"].transform(input_df[feature_names])
    log_predictions = np.asarray(model_bundle["model"].predict(X_prepared))

    if log_predictions.ndim == 1:
        log_predictions = log_predictions.reshape(-1, len(quantiles))

    price_predictions = np.expm1(log_predictions)
    price_predictions = np.clip(price_predictions, 0, None)
    price_predictions = np.maximum.accumulate(price_predictions, axis=1)

    columns = [f"price_p{int(q * 100):02d}" for q in quantiles]
    return pd.DataFrame(price_predictions, columns=columns, index=input_df.index)


def _select_interval_columns(predictions, preferred_interval):
    """Choose lower and upper prediction columns for the dashboard range."""
    lower_col, upper_col = INTERVAL_COLUMNS.get(preferred_interval, INTERVAL_COLUMNS["P05-P95"])

    if lower_col in predictions.index and upper_col in predictions.index:
        return lower_col, upper_col

    for fallback_interval in ("P05-P95", "P10-P90"):
        lower_col, upper_col = INTERVAL_COLUMNS[fallback_interval]
        if lower_col in predictions.index and upper_col in predictions.index:
            return lower_col, upper_col

    quantile_cols = sorted(col for col in predictions.index if col.startswith("price_p"))
    if len(quantile_cols) < 3:
        raise ValueError("The saved model did not return enough quantile predictions.")

    return quantile_cols[0], quantile_cols[-1]


def predict_price(user_input, model_path=DEFAULT_MODEL_PATH):
    """Predict lower, median, and upper dashboard price estimates.

    Parameters
    ----------
    user_input : dict
        Dashboard input values keyed by the CatBoost model feature names.
    model_path : str or pathlib.Path, optional
        Path to the saved model bundle.

    Returns
    -------
    tuple[float, float, float]
        Lower quantile estimate, median estimate, and upper quantile estimate.
    """
    model_bundle = load_price_model(Path(model_path))
    input_df = _prepare_prediction_input(user_input)
    predictions = _predict_quantiles(model_bundle, input_df).iloc[0]

    preferred_interval = model_bundle.get("preferred_dashboard_interval", "P05-P95")
    lower_col, upper_col = _select_interval_columns(predictions, preferred_interval)

    lower = float(predictions[lower_col])
    median = float(predictions["price_p50"])
    upper = float(predictions[upper_col])

    return lower, median, upper
