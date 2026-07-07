# Australian Used Car Price Prediction

## Project Overview

This project builds an end-to-end machine learning workflow for estimating Australian used-car listing prices from vehicle and listing characteristics.

The repository includes:

* A full data science notebook covering cleaning, EDA, feature engineering, model comparison, final CatBoost evaluation, residual analysis, feature importance, and sample predictions.
* A CatBoost multi-quantile training notebook for dashboard price intervals.
* A saved dashboard model artifact.
* A Dash app that returns a predicted market range, a median estimate, and comparable real listings.

The project focuses on typical consumer-facing used-car valuations. Extreme high-price listings above the 99th percentile are excluded from modelling so the model is not dominated by exotic or unusually expensive vehicles.

---

## Dataset

The dataset contains Australian vehicle listings from 2023 and is stored in `data/`.

Key variables include:

* Brand, model, and year
* Kilometres
* Transmission, fuel type, and drive type
* Engine size, fuel consumption, cylinders, doors, and seats
* Body type and exterior colour
* Region and state
* Listing title
* Price

Only used vehicles are retained for modelling. Missing target values are removed, one anomalous `$88` listing is removed, and the modelling dataset is restricted to listings up to the 99th percentile of price.

---

## Modelling Workflow

The main notebook performs:

1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Train/test splitting
5. Baseline model comparison
6. Final CatBoost point-estimate evaluation
7. Residual analysis
8. Feature importance analysis
9. Sample predictions
10. Export of the exact modelling dataset and split metadata

The exported modelling files are then reused by the interval-model notebook so the dashboard model is trained on the same cleaned dataframe, feature set, and train/test split.

---

## Models

The main notebook compares:

* Ridge Regression
* Random Forest Regression
* CatBoost Regression

The final point-estimate model is a CatBoost regression model trained with a log-transformed target. CatBoost was selected because it performed strongly and handles categorical predictors well, including the high-cardinality `model` feature.

Held-out test results for the point-estimate CatBoost model:

* MAE: approximately `$3,841`
* MAPE: `12.98%`
* RMSE: approximately `$6,321`
* R-squared: `0.897`

For the dashboard, a separate CatBoost `MultiQuantile` model is trained to return multiple price quantiles. The dashboard uses this model to provide a median estimate and a broad market range rather than a single fixed price.

---

## Dashboard

The Dash dashboard is located in `dashboard/app.py`.

It provides:

* Searchable dropdowns populated from the cleaned modelling dataset.
* Dependent dropdown filtering for brand, model, body type, fuel type, and drive type.
* Median-based autofill for technical vehicle fields such as engine size, fuel consumption, cylinders, doors, and seats.
* CatBoost quantile predictions for the selected vehicle.
* A market comparison using nearest-neighbour comparable listings.
* A visual comparison of the model range against similar-listing prices.

Run the dashboard from the project root:

```bash
python dashboard/app.py
```

---

## Repository Contents

```text
used-car-price-prediction/
|-- dashboard/
|   |-- app.py
|   `-- README.md
|-- data/
|   |-- australian_vehicle_prices_2023.csv
|   `-- model_inputs/
|       |-- used_car_modelling_dataset.csv
|       |-- used_car_modelling_metadata.json
|       |-- used_car_train_index.csv
|       `-- used_car_test_index.csv
|-- models/
|   `-- catboost_used_car_price_quantile_model.pkl
|-- notebooks/
|   |-- used_car_price_prediction_australia.ipynb
|   `-- catboost_multiquantile_model.ipynb
|-- src/
|   |-- prediction.py
|   `-- similar_listings.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

---

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The main dashboard dependencies are Dash, Dash Bootstrap Components, pandas, scikit-learn, joblib, and CatBoost.

---

## Key Findings

* `year` and `kilometres` are the strongest predictors of used-car price.
* `brand`, `drive_type`, `fuel_type`, and `model` also provide important pricing information.
* Used-car prices are strongly right-skewed, supporting the use of a log-transformed target for point-estimate modelling.
* Prediction errors are generally centred around zero, but higher-priced vehicles show greater error variability.
* Important details such as trim, badge, variant, condition, service history, modifications, and optional extras are not available as clean structured variables.

---

## Future Improvements

Potential future extensions include:

* Extract structured trim, badge, and variant information from the `title` column.
* Improve inconsistent categorical labels, especially where model names are broad or ambiguous.
* Add richer vehicle condition, service history, accident history, modifications, and optional extras if available.
* Extend the dataset beyond 2023 listings.
* Explore periodic data collection and model retraining so the dashboard reflects newer market conditions.
* Add curated vehicle preview images to the dashboard using local, licensed image assets.
