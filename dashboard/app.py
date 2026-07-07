"""Dash application for the Australian used-car price estimator."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELLING_DATA_PATH = PROJECT_ROOT / "data" / "model_inputs" / "used_car_modelling_dataset.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html

from src.prediction import predict_price
from src.similar_listings import find_similar_listings


NUMERIC_INPUTS = [
    ("year", "Year", 2020),
    ("kilometres", "Kilometres", 65000),
    ("engine_size_litres", "Engine Size (L)", 2.0),
    ("fuel_consumption_per_100km", "Fuel Consumption (L/100 km)", 7.5),
    ("cylinders", "Cylinders", 4),
    ("doors", "Doors", 4),
    ("seats", "Seats", 5),
]

CATEGORICAL_INPUTS = [
    ("brand", "Brand"),
    ("model", "Model"),
    ("transmission", "Transmission"),
    ("body_type", "Body Type"),
    ("fuel_type", "Fuel Type"),
    ("drive_type", "Drive Type"),
    ("state", "State"),
    ("colour_ext", "Exterior Colour"),
]

INPUT_LAYOUT_ORDER = [
    "brand",
    "model",
    "year",
    "kilometres",
    "transmission",
    "body_type",
    "fuel_type",
    "drive_type",
    "engine_size_litres",
    "fuel_consumption_per_100km",
    "cylinders",
    "doors",
    "seats",
    "state",
    "colour_ext",
]

FALLBACK_OPTIONS = {
    "brand": ["Toyota", "Mazda", "Ford", "Hyundai", "BMW"],
    "model": ["Corolla", "CX-5", "Ranger", "i30", "3"],
    "transmission": ["Automatic", "Manual"],
    "body_type": ["SUV", "Hatchback", "Sedan", "Ute / Tray", "Wagon"],
    "fuel_type": ["Unleaded", "Diesel", "Premium", "Hybrid", "Electric"],
    "drive_type": ["Front", "Rear", "AWD", "4WD"],
    "state": ["NSW", "VIC", "QLD", "SA", "WA", "ACT", "TAS", "NT"],
    "colour_ext": ["White", "Black", "Grey", "Silver", "Blue", "Red"],
}

AUTOFILL_NUMERIC_INPUTS = [
    "engine_size_litres",
    "fuel_consumption_per_100km",
    "cylinders",
    "doors",
    "seats",
]

DRIVE_TYPE_LABELS = {
    "Front": "FWD",
    "Rear": "RWD",
}

SIMILAR_LISTING_COLUMNS = [
    {"name": "Title", "id": "title"},
    {"name": "Kilometres", "id": "kilometres"},
    {"name": "Price", "id": "price"},
    {"name": "Location", "id": "location"},
    {"name": "Transmission", "id": "transmission"},
    {"name": "Fuel Type", "id": "fuel_type"},
    {"name": "Body Type", "id": "body_type"},
    {"name": "Engine Size (L)", "id": "engine_size_litres"},
    {"name": "Cylinders", "id": "cylinders"},
]


def load_listings_data():
    """Load the cleaned modelling listings used by the dashboard."""
    if not MODELLING_DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(MODELLING_DATA_PATH)


def make_options(values, column=None):
    """Convert raw values into Dash dropdown options."""
    options = []

    for value in values:
        label = DRIVE_TYPE_LABELS.get(
            value, value) if column == "drive_type" else value
        options.append({"label": label, "value": value})

    return options


def filter_listings(filters):
    """Filter the cleaned listings by selected categorical values."""
    filtered = listings_df

    if filtered.empty:
        return filtered

    for col, value in filters.items():
        if value is None or col not in filtered.columns:
            continue
        filtered = filtered[filtered[col].astype(str) == str(value)]

    return filtered


def get_dropdown_options(column, filters=None):
    """Return valid dropdown options for a column after applying filters."""
    filters = filters or {}
    filtered = filter_listings(filters)

    if filtered.empty or column not in filtered.columns:
        values = FALLBACK_OPTIONS[column]
    else:
        values = (
            filtered[column]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        values = values if values else FALLBACK_OPTIONS[column]

    return make_options(values, column)


def choose_dropdown_value(options, current_value=None):
    """Keep the current value when valid, otherwise use the first available option."""
    valid_values = {option["value"] for option in options}

    if current_value in valid_values:
        return current_value

    return options[0]["value"] if options else None


def get_autofill_numeric_values(brand, model, body_type):
    """Use listing medians to pre-fill hard-to-know numeric vehicle attributes."""
    fallback_values = {
        input_id: value
        for input_id, _, value in NUMERIC_INPUTS
        if input_id in AUTOFILL_NUMERIC_INPUTS
    }

    filter_steps = [
        {"brand": brand, "model": model, "body_type": body_type},
        {"brand": brand, "model": model},
        {"brand": brand},
    ]

    for filters in filter_steps:
        filtered = filter_listings(filters)
        if filtered.empty:
            continue

        medians = filtered[AUTOFILL_NUMERIC_INPUTS].apply(
            pd.to_numeric, errors="coerce"
        ).median()

        values = {}
        for input_id in AUTOFILL_NUMERIC_INPUTS:
            if pd.isna(medians[input_id]):
                values[input_id] = fallback_values[input_id]
            elif input_id in {"cylinders", "doors", "seats"}:
                values[input_id] = int(round(float(medians[input_id])))
            else:
                values[input_id] = round(float(medians[input_id]), 1)

        return values

    return fallback_values


def load_dropdown_options(listings):
    """Load dropdown options from the exported modelling dataset when available."""
    options = {}

    for col, fallback_values in FALLBACK_OPTIONS.items():
        if listings.empty or col not in listings.columns:
            values = fallback_values
        else:
            values = (
                listings[col]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            values = values if values else fallback_values

        options[col] = make_options(values, col)

    return options


def format_currency(value):
    """Format a numeric value as Australian dollars."""
    if pd.isna(value):
        return "Not available"
    return f"${float(value):,.0f}"


def format_number(value):
    """Format a numeric value without decimals."""
    if pd.isna(value):
        return "Not available"
    return f"{float(value):,.0f}"


def get_percent_position(value, axis_min, axis_max):
    """Convert a price value into a bounded 0-100 position."""
    if pd.isna(value) or axis_max <= axis_min:
        return 50

    position = ((value - axis_min) / (axis_max - axis_min)) * 100
    return max(0, min(100, position))


def make_numeric_input(input_id, label, value):
    """Create a labelled numeric input."""
    return dbc.Col(
        [
            dbc.Label(label, html_for=input_id,
                      className="fw-semibold small text-muted"),
            dbc.Input(
                id=input_id,
                type="number",
                value=value,
                debounce=True,
                className="shadow-sm",
            ),
        ],
        xs=12,
        md=6,
        lg=4,
        className="mb-3",
    )


def make_dropdown(input_id, label, options):
    """Create a labelled searchable dropdown."""
    default_value = options[0]["value"] if options else None

    return dbc.Col(
        [
            dbc.Label(label, html_for=input_id,
                      className="fw-semibold small text-muted"),
            dcc.Dropdown(
                id=input_id,
                options=options,
                value=default_value,
                searchable=True,
                clearable=False,
                placeholder=f"Select {label.lower()}",
                className="shadow-sm",
            ),
        ],
        xs=12,
        md=6,
        lg=4,
        className="mb-3",
    )


def make_kpi_card(title, value, subtitle, value_id, subtitle_id):
    """Create a compact result KPI block."""
    return html.Div(
        [
            html.Div(
                title, className="text-uppercase text-muted small fw-semibold"),
            html.Div(value, id=value_id,
                     className="display-6 fw-bold text-dark mt-1"),
            html.Div(subtitle, id=subtitle_id,
                     className="text-muted small mt-2"),
        ],
        className="model-kpi-block h-100",
    )


def make_vehicle_input(input_id, numeric_inputs, categorical_inputs, dropdown_options):
    """Create the correct input component for a vehicle feature."""
    if input_id in numeric_inputs:
        label, value = numeric_inputs[input_id]
        return make_numeric_input(input_id, label, value)

    label = categorical_inputs[input_id]
    return make_dropdown(input_id, label, dropdown_options.get(input_id, []))


def make_range_visual_row(label, low, high, point, point_label, axis_min, axis_max, range_class, point_class):
    """Create one horizontal market range row."""
    low_position = get_percent_position(low, axis_min, axis_max)
    high_position = get_percent_position(high, axis_min, axis_max)
    point_position = get_percent_position(point, axis_min, axis_max)
    range_width = max(1, high_position - low_position)

    return html.Div(
        [
            html.Div(label, className="fw-semibold mb-2"),
            html.Div(
                [
                    html.Div(
                        format_currency(low),
                        className="market-end-label market-low-label",
                        style={"left": f"{low_position}%"},
                    ),
                    html.Div(
                        format_currency(high),
                        className="market-end-label market-high-label",
                        style={"left": f"{high_position}%"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                className=f"market-range {range_class}",
                                style={"left": f"{low_position}%",
                                       "width": f"{range_width}%"},
                            ),
                            html.Div(
                                className=f"market-point {point_class}",
                                title=f"{point_label}: {format_currency(point)}",
                                style={"left": f"{point_position}%"},
                            ),
                        ],
                        className="market-track",
                    ),
                ],
                className="market-plot",
            ),
        ],
        className="market-range-row mb-4",
    )


def make_similar_listing_summary(similar_prices):
    """Create the comparable-listing price summary."""
    if similar_prices.empty:
        return html.Div(
            "Run a prediction to compare against similar real listing prices.",
            className="text-muted",
        )

    min_price = similar_prices.min()
    max_price = similar_prices.max()
    median_price = similar_prices.median()
    average_price = similar_prices.mean()

    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Div("Similar Listings Range",
                             className="text-uppercase text-muted small fw-semibold"),
                    html.Div(
                        f"{format_currency(min_price)} - {format_currency(max_price)}",
                        className="h3 fw-bold mb-2",
                    ),
                    html.Div(
                        "Lowest to highest price among the comparable listings below.",
                        className="text-muted",
                    ),
                ],
                xs=12,
                lg=4,
                className="mb-3",
            ),
            dbc.Col(
                [
                    html.Div("Similar Listings Median",
                             className="text-uppercase text-muted small fw-semibold"),
                    html.Div(format_currency(median_price),
                             className="h3 fw-bold text-dark mb-2"),
                    html.Div("Middle price of the comparable listings.",
                             className="text-muted"),
                ],
                xs=12,
                lg=4,
                className="mb-3",
            ),
            dbc.Col(
                [
                    html.Div("Similar Listings Average",
                             className="text-uppercase text-muted small fw-semibold"),
                    html.Div(format_currency(average_price),
                             className="h3 fw-bold text-dark mb-2"),
                    html.Div("Mean price of the comparable listings.",
                             className="text-muted"),
                ],
                xs=12,
                lg=4,
                className="mb-3",
            ),
        ]
    )


def make_market_interval_visual(similar_prices, model_lower=None, model_median=None, model_upper=None):
    """Create the model versus similar-listings interval visual."""
    if similar_prices.empty or pd.isna(model_lower) or pd.isna(model_upper) or pd.isna(model_median):
        return html.Div(
            "Run a prediction to view the model range against comparable listing prices.",
            className="text-muted",
        )

    min_price = similar_prices.min()
    max_price = similar_prices.max()
    median_price = similar_prices.median()
    axis_min = min(model_lower, min_price)
    axis_max = max(model_upper, max_price)

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        [html.Span(className="legend-swatch model-swatch"), "Model range"]),
                    html.Span(
                        [html.Span(className="legend-swatch similar-swatch"), "Similar listings range"]),
                    html.Span([html.Span(className="legend-dot"), "Median"]),
                ],
                className="market-legend mb-4",
            ),
            make_range_visual_row(
                "Model predicted range",
                model_lower,
                model_upper,
                model_median,
                "Model median estimate",
                axis_min,
                axis_max,
                "model-range",
                "model-point",
            ),
            make_range_visual_row(
                "Similar listings price range",
                min_price,
                max_price,
                median_price,
                "Similar listings median",
                axis_min,
                axis_max,
                "similar-range",
                "similar-point",
            ),
        ]
    )


def collect_user_input(values):
    """Map callback values into the dictionary expected by helper functions."""
    return dict(zip(INPUT_LAYOUT_ORDER, values))


def format_similar_listings(similar_df):
    """Format similar listings for display in the dashboard table."""
    display_df = similar_df.copy()

    if display_df.empty:
        return []

    if "kilometres" in display_df:
        display_df["kilometres"] = pd.to_numeric(
            display_df["kilometres"], errors="coerce"
        ).apply(format_number)

    if "price" in display_df:
        display_df["price"] = pd.to_numeric(
            display_df["price"], errors="coerce"
        ).apply(format_currency)

    return display_df.to_dict("records")


listings_df = load_listings_data()
dropdown_options = load_dropdown_options(listings_df)
numeric_input_lookup = {
    input_id: (label, value)
    for input_id, label, value in NUMERIC_INPUTS
}
categorical_input_lookup = {
    input_id: label
    for input_id, label in CATEGORICAL_INPUTS
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        html.Div(
            [
                html.H1("Used Car Price Estimator", className="fw-bold mb-2"),
                html.P(
                    (
                        "Estimate a used-car market price range using a CatBoost "
                        "quantile model and comparable listings."
                    ),
                    className="lead text-muted mb-0",
                ),
            ],
            className="py-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H2("Vehicle Details",
                                    className="h4 fw-bold mb-1"),
                            html.P(
                                "Enter the vehicle characteristics used by the pricing model.",
                                className="text-muted mb-4",
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            make_vehicle_input(
                                input_id,
                                numeric_input_lookup,
                                categorical_input_lookup,
                                dropdown_options,
                            )
                            for input_id in INPUT_LAYOUT_ORDER
                        ]
                    ),
                    html.Div(
                        dbc.Button(
                            "Predict Price",
                            id="predict-price-button",
                            color="primary",
                            size="lg",
                            className="predict-button px-5 py-3",
                        ),
                        className="text-center mt-3",
                    ),
                ]
            ),
            className="border-0 shadow-sm mb-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Model Outputs", className="h4 fw-bold mb-1"),
                    html.P(
                        "Estimated market range for the selected vehicle.",
                        className="text-muted mb-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                make_kpi_card(
                                    "Predicted Price Range",
                                    "Not run yet",
                                    "Run a prediction to estimate the market price range.",
                                    "predicted-range-value",
                                    "predicted-range-subtitle",
                                ),
                                xs=12,
                                lg=6,
                                className="mb-3 mb-lg-0",
                            ),
                            dbc.Col(
                                make_kpi_card(
                                    "Median Estimate",
                                    "Not run yet",
                                    "Run a prediction to estimate the median price.",
                                    "median-estimate-value",
                                    "median-estimate-subtitle",
                                ),
                                xs=12,
                                lg=6,
                            ),
                        ],
                    ),
                ]
            ),
            className="model-output-card border-0 shadow-sm mb-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Market Comparison", className="h4 fw-bold mb-3"),
                    html.Div(id="price-comparison-content",
                             children=make_similar_listing_summary(pd.Series(dtype=float))),
                ]
            ),
            className="border-0 shadow-sm mb-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Model Range vs Similar Listings",
                            className="h4 fw-bold mb-3"),
                    html.Div(
                        id="market-interval-visual",
                        children=make_market_interval_visual(
                            pd.Series(dtype=float)),
                    ),
                ]
            ),
            className="border-0 shadow-sm mb-4",
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.H2("Similar Listings",
                                    className="h4 fw-bold mb-1"),
                            html.P(
                                "Comparable real listings selected using the dashboard input values.",
                                className="text-muted mb-3",
                            ),
                        ]
                    ),
                    dash_table.DataTable(
                        id="similar-listings-table",
                        data=[],
                        columns=SIMILAR_LISTING_COLUMNS,
                        page_size=5,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif",
                            "fontSize": "14px",
                            "padding": "12px",
                            "textAlign": "left",
                        },
                        style_header={
                            "backgroundColor": "#f8f9fa",
                            "fontWeight": "700",
                            "borderBottom": "1px solid #dee2e6",
                        },
                        style_data={
                            "borderBottom": "1px solid #f1f3f5",
                        },
                    ),
                ]
            ),
            className="border-0 shadow-sm mb-5",
        ),
    ],
    fluid=True,
    className="dashboard-page px-3 px-md-5",
)


@app.callback(
    Output("model", "options"),
    Output("model", "value"),
    Input("brand", "value"),
    State("model", "value"),
)
def update_model_options(brand, current_model):
    """Limit model options to vehicles from the selected brand."""
    options = get_dropdown_options("model", {"brand": brand})
    return options, choose_dropdown_value(options, current_model)


@app.callback(
    [
        Output("body_type", "options"),
        Output("body_type", "value"),
        Output("fuel_type", "options"),
        Output("fuel_type", "value"),
        Output("drive_type", "options"),
        Output("drive_type", "value"),
    ],
    Input("brand", "value"),
    Input("model", "value"),
    State("body_type", "value"),
    State("fuel_type", "value"),
    State("drive_type", "value"),
)
def update_vehicle_category_options(
    brand,
    model,
    current_body_type,
    current_fuel_type,
    current_drive_type,
):
    """Limit category options to combinations found for the selected brand/model."""
    filters = {"brand": brand, "model": model}

    body_options = get_dropdown_options("body_type", filters)
    fuel_options = get_dropdown_options("fuel_type", filters)
    drive_options = get_dropdown_options("drive_type", filters)

    return (
        body_options,
        choose_dropdown_value(body_options, current_body_type),
        fuel_options,
        choose_dropdown_value(fuel_options, current_fuel_type),
        drive_options,
        choose_dropdown_value(drive_options, current_drive_type),
    )


@app.callback(
    [
        Output("engine_size_litres", "value"),
        Output("fuel_consumption_per_100km", "value"),
        Output("cylinders", "value"),
        Output("doors", "value"),
        Output("seats", "value"),
    ],
    Input("brand", "value"),
    Input("model", "value"),
    Input("body_type", "value"),
)
def autofill_numeric_vehicle_details(brand, model, body_type):
    """Pre-fill technical numeric fields from comparable listing medians."""
    values = get_autofill_numeric_values(brand, model, body_type)

    return (
        values["engine_size_litres"],
        values["fuel_consumption_per_100km"],
        values["cylinders"],
        values["doors"],
        values["seats"],
    )


@app.callback(
    [
        Output("predicted-range-value", "children"),
        Output("predicted-range-subtitle", "children"),
        Output("median-estimate-value", "children"),
        Output("median-estimate-subtitle", "children"),
        Output("price-comparison-content", "children"),
        Output("market-interval-visual", "children"),
        Output("similar-listings-table", "data"),
    ],
    Input("predict-price-button", "n_clicks"),
    [State(input_id, "value") for input_id in INPUT_LAYOUT_ORDER],
    prevent_initial_call=True,
)
def update_dashboard_outputs(n_clicks, *values):
    """Run the prediction and similar-listings lookup from dashboard inputs."""
    user_input = collect_user_input(values)

    lower, median, upper = predict_price(user_input)

    similar_df = find_similar_listings(user_input, listings_df, n=5)
    similar_prices = pd.to_numeric(
        similar_df.get("price", pd.Series(dtype=float)), errors="coerce"
    ).dropna()

    return (
        f"{format_currency(lower)} - {format_currency(upper)}",
        "Estimated market price range.",
        format_currency(median),
        "Median predicted price.",
        make_similar_listing_summary(similar_prices),
        make_market_interval_visual(similar_prices, lower, median, upper),
        format_similar_listings(similar_df),
    )


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Used Car Price Estimator</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: #f4f6f8;
                color: #1f2933;
            }

            .dashboard-page {
                max-width: 1320px;
            }

            .Select-control,
            .form-control {
                border-color: #d7dde5;
                min-height: 42px;
            }

            .predict-button {
                min-width: 240px;
            }

            .model-output-card {
                background: #f9fafb;
            }

            .model-kpi-block {
                background: #ffffff;
                border: 1px solid #edf0f3;
                border-radius: 8px;
                padding: 1.25rem;
            }

            .market-legend {
                display: flex;
                flex-wrap: wrap;
                gap: 10px 18px;
                color: #5b6573;
                font-size: 0.95rem;
            }

            .market-legend span {
                align-items: center;
                display: inline-flex;
                gap: 7px;
            }

            .legend-swatch {
                border-radius: 999px;
                height: 8px;
                width: 28px;
            }

            .model-swatch {
                background: #0d6efd;
            }

            .similar-swatch {
                background: #20c997;
            }

            .legend-dot {
                background: #495057;
                border: 2px solid #ffffff;
                border-radius: 50%;
                box-shadow: 0 1px 4px rgba(31, 41, 51, 0.2);
                height: 13px;
                width: 13px;
            }

            .market-plot {
                padding-top: 26px;
                position: relative;
            }

            .market-end-label {
                color: #374151;
                font-size: 0.95rem;
                font-weight: 600;
                position: absolute;
                top: 0;
                white-space: nowrap;
            }

            .market-low-label {
                transform: translateX(0);
            }

            .market-high-label {
                transform: translateX(-100%);
            }

            .market-track {
                position: relative;
                height: 18px;
                border-radius: 999px;
                background: #edf1f5;
            }

            .market-range {
                position: absolute;
                top: 5px;
                height: 8px;
                border-radius: 999px;
            }

            .model-range {
                background: #0d6efd;
            }

            .similar-range {
                background: #20c997;
            }

            .market-point {
                position: absolute;
                top: 50%;
                width: 16px;
                height: 16px;
                border: 3px solid #ffffff;
                border-radius: 50%;
                box-shadow: 0 1px 5px rgba(31, 41, 51, 0.25);
                transform: translate(-50%, -50%);
            }

            .model-point {
                background: #0b5ed7;
            }

            .similar-point {
                background: #158765;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
