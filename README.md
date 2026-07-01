# Australian Used Car Price Prediction

## Project Overview

This project develops a machine learning workflow for estimating Australian used-car listing prices using vehicle and listing characteristics.

The project includes data cleaning, exploratory data analysis, feature engineering, model comparison, final model evaluation, residual analysis, and model interpretation. A key focus of the project is building a model that can support a future dashboard-based valuation tool for typical used-car listings.

Because vehicle prices are strongly right-skewed, the modelling stage uses a log-transformed target variable so that prediction errors are treated more consistently across lower- and higher-priced vehicles.

---

## Dataset

The dataset contains Australian vehicle listings from 2023 and is included in the `data/` directory.

Variables include:

* Brand, model, and year
* Kilometres
* Transmission
* Fuel type
* Drive type
* Engine size
* Fuel consumption
* Cylinders, doors, and seats
* Body type
* Exterior colour
* Location information
* Price

The modelling dataset was restricted to used vehicles and filtered to retain listings up to the 99th percentile of price to focus on the main used-car market rather than exotic or extreme luxury listings.

---

## Project Workflow

1. Data cleaning and preprocessing
2. Feature engineering
3. Exploratory data analysis
4. Target transformation
5. Train/test splitting
6. Baseline model comparison
7. CatBoost model evaluation
8. Residual analysis
9. Feature importance analysis
10. Sample predictions and interpretation

---

## Models Evaluated

The following regression models were compared:

* Ridge Regression
* Random Forest Regression
* CatBoost Regression

Model performance was assessed using cross-validation and evaluated with:

* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)
* Root Mean Squared Error (RMSE)
* R-squared

---

## Final Model

The final selected model was a **CatBoost Regression** model trained using a log-transformed price target.

CatBoost was selected because it achieved the strongest overall performance and can handle categorical predictors directly, including higher-cardinality variables such as `model`.

On the held-out test set, the model achieved:

* MAE: approximately $3,841
* MAPE: 12.98%
* RMSE: approximately $6,321
* R-squared: 0.897

---

## Key Findings

* `year` and `kilometres` were the strongest predictors of used-car price.
* `brand`, `drive_type`, and `fuel_type` also provided important pricing information.
* Used-car prices were strongly right-skewed, supporting the use of a log-transformed target.
* Prediction errors were generally centred around zero, but error variability increased for higher-priced vehicles.
* Some important vehicle details, such as trim, badge, variant, condition, modifications, and service history, were not available as clean structured variables.

---

## Repository Contents

* `used_car_price_prediction_australia.ipynb` — Complete analysis, modelling workflow, evaluation, and interpretation.
* `dashboard/` — Placeholder for the future interactive dashboard implementation.
* `data/` — Dataset used in the project.
* `models/` — Folder reserved for trained dashboard model files.

---

## Future Improvements

Potential future extensions include:

* Train a quantile CatBoost model for the dashboard to provide a median estimate and valuation range.
* Extract trim, badge, and variant information from the `title` column using more advanced text cleaning.
* Improve inconsistent categorical labels, especially where model names are too broad.
* Add vehicle condition, service history, accident history, modifications, and optional extras if available.
* Extend the dataset beyond 2023 listings.
* Explore automated scraping or periodic data collection from car listing websites to keep the model updated.
* Build an interactive dashboard for user-entered vehicle details.
