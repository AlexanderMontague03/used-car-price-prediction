# Used Car Price Estimator Dashboard

This folder contains the Dash application for the Australian used-car price estimator.

The dashboard is designed as a consumer-facing valuation tool. It combines a trained CatBoost quantile model with comparable real listings so users can see both a modelled price range and nearby market examples.

## What The Dashboard Shows

The app currently provides:

* A vehicle input form using the same feature set as the trained model.
* Searchable dropdowns populated from the cleaned modelling dataset.
* Dependent dropdown filtering for brand, model, body type, fuel type, and drive type.
* Median-based autofill for technical fields such as engine size, fuel consumption, cylinders, doors, and seats.
* A predicted market price range and median estimate from the CatBoost multi-quantile model.
* Comparable-listing summary statistics based on similar real listings.
* A visual comparison of the model range against similar-listing prices.
* A table of nearest-neighbour comparable listings.

## Files Used By The App

The dashboard loads:

* `models/catboost_used_car_price_quantile_model.pkl` for model predictions.
* `data/model_inputs/used_car_modelling_dataset.csv` for dropdown values and comparable-listing search.
* `src/prediction.py` for model-loading and prediction helper logic.
* `src/similar_listings.py` for nearest-neighbour comparable-listing logic.

## Running The Dashboard

From the project root, install dependencies if needed:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python dashboard/app.py
```

The app should open locally at:

```text
http://127.0.0.1:8050/
```

If Python reports `No module named 'src'`, make sure the command is being run from the project root and that the repository still contains the `src/` folder.

## Notes

The dashboard does not retrain the model. It loads the saved model artifact from `models/` and uses the exported modelling dataset only for valid input options and similar-listing comparisons.

The similar-listing search is interpretability support only. It does not adjust or train the CatBoost model.

## Future Dashboard Improvements

Possible dashboard enhancements include:

* Add a selected-vehicle preview panel with a local, licensed example image when a brand, model, and year are selected.
* Add clearer fallback messaging when there are very few comparable listings for a selected vehicle.
* Add optional filters for state, kilometres, or transmission within the comparable-listings table.
* Improve mobile layout spacing for smaller screens.
* Add a short interpretation message explaining whether the model estimate is aligned with similar-listing prices.
* Add deployment configuration for a hosted dashboard demo.
