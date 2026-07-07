"""Similar-listing helpers for the used-car price dashboard."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


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

DISPLAY_COLUMNS = [
    "title",
    "kilometres",
    "price",
    "location",
    "transmission",
    "fuel_type",
    "body_type",
    "engine_size_litres",
    "cylinders",
]


def _normalise_text(value):
    """Normalise text values for robust filtering."""
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _filter_by_year_window(candidate_df, user_input, n):
    """Prefer same-year or nearby-year listings before running KNN."""
    if "year" not in candidate_df.columns:
        return candidate_df

    target_year = pd.to_numeric(pd.Series([user_input.get("year")]), errors="coerce").iloc[0]
    if pd.isna(target_year):
        return candidate_df

    listing_years = pd.to_numeric(candidate_df["year"], errors="coerce")

    for year_window in (0, 1, 2, 3, 5):
        if year_window == 0:
            year_mask = listing_years == target_year
        else:
            year_mask = (listing_years - target_year).abs() <= year_window

        year_candidates = candidate_df[year_mask].copy()
        if len(year_candidates) >= n:
            return year_candidates

    return candidate_df


def _filter_candidate_listings(user_input, listings_df, n):
    """Filter to the same brand/model, then prefer same or nearby listing years."""
    brand = _normalise_text(user_input.get("brand"))
    model = _normalise_text(user_input.get("model"))

    brand_values = listings_df["brand"].map(_normalise_text) if "brand" in listings_df else pd.Series("", index=listings_df.index)
    model_values = listings_df["model"].map(_normalise_text) if "model" in listings_df else pd.Series("", index=listings_df.index)

    same_brand_model = listings_df[(brand_values == brand) & (model_values == model)].copy()
    if not same_brand_model.empty:
        return _filter_by_year_window(same_brand_model, user_input, n)

    # If the selected model is unavailable in the cleaned data, fall back to
    # same-brand listings so the dashboard can still return an interpretable table.
    same_brand = listings_df[brand_values == brand].copy()
    if not same_brand.empty:
        return _filter_by_year_window(same_brand, user_input, n)

    return _filter_by_year_window(listings_df.copy(), user_input, n)


def _prepare_numeric_features(candidate_df, input_df):
    """Impute and scale numeric features using candidate-listing statistics."""
    candidate_numeric = candidate_df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    input_numeric = input_df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")

    medians = candidate_numeric.median()
    candidate_numeric = candidate_numeric.fillna(medians)
    input_numeric = input_numeric.fillna(medians)

    means = candidate_numeric.mean()
    stds = candidate_numeric.std(ddof=0).replace(0, 1).fillna(1)

    candidate_scaled = (candidate_numeric - means) / stds
    input_scaled = (input_numeric - means) / stds

    return candidate_scaled.reset_index(drop=True), input_scaled.reset_index(drop=True)


def _prepare_categorical_features(candidate_df, input_df):
    """One-hot encode categorical features with shared candidate/input columns."""
    candidate_categorical = (
        candidate_df[CATEGORICAL_FEATURES]
        .astype("object")
        .where(pd.notna(candidate_df[CATEGORICAL_FEATURES]), "Missing")
        .astype(str)
    )
    input_categorical = (
        input_df[CATEGORICAL_FEATURES]
        .astype("object")
        .where(pd.notna(input_df[CATEGORICAL_FEATURES]), "Missing")
        .astype(str)
    )

    combined = pd.concat([candidate_categorical, input_categorical], ignore_index=True)
    encoded = pd.get_dummies(combined, columns=CATEGORICAL_FEATURES, dtype=float)

    candidate_encoded = encoded.iloc[: len(candidate_df)].reset_index(drop=True)
    input_encoded = encoded.iloc[[len(candidate_df)]].reset_index(drop=True)

    return candidate_encoded, input_encoded


def _prepare_knn_matrices(candidate_df, user_input):
    """Create scaled numeric and encoded categorical matrices for nearest neighbours."""
    input_row = {feature: user_input.get(feature, np.nan) for feature in MODEL_FEATURES}
    input_df = pd.DataFrame([input_row], columns=MODEL_FEATURES)

    candidate_features = candidate_df[MODEL_FEATURES].copy()

    candidate_numeric, input_numeric = _prepare_numeric_features(candidate_features, input_df)
    candidate_categorical, input_categorical = _prepare_categorical_features(candidate_features, input_df)

    X_candidates = pd.concat([candidate_numeric, candidate_categorical], axis=1)
    X_input = pd.concat([input_numeric, input_categorical], axis=1)

    return X_candidates, X_input


def _add_display_location(display_df, source_df):
    """Create a display location from region/state when no location column exists."""
    if "location" in source_df.columns:
        display_df["location"] = source_df["location"].values
        return display_df

    if {"region", "state"}.issubset(source_df.columns):
        display_df["location"] = (
            source_df["region"].fillna("").astype(str).str.strip() +
            ", " +
            source_df["state"].fillna("").astype(str).str.strip()
        ).str.strip(", ")
        return display_df

    display_df["location"] = np.nan
    return display_df


def find_similar_listings(user_input, listings_df, n=5):
    """Return the closest real listings for dashboard interpretation.

    This nearest-neighbour search is only for displaying comparable listings.
    It is not used to train or adjust the CatBoost prediction model.
    """
    if listings_df.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    candidate_df = _filter_candidate_listings(user_input, listings_df, n)
    candidate_df = candidate_df.dropna(how="all").copy()

    if candidate_df.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    n_neighbors = min(n, len(candidate_df))
    X_candidates, X_input = _prepare_knn_matrices(candidate_df, user_input)

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X_candidates)
    _, neighbour_positions = knn.kneighbors(X_input)

    similar = candidate_df.iloc[neighbour_positions[0]].copy()

    display_df = pd.DataFrame(index=similar.index)
    for col in DISPLAY_COLUMNS:
        if col == "location":
            continue
        display_df[col] = similar[col] if col in similar.columns else np.nan

    display_df = _add_display_location(display_df, similar)
    return display_df[DISPLAY_COLUMNS].reset_index(drop=True)
