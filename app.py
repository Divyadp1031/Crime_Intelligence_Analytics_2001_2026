"""
CrimeLens - Flask Application Entry Point
================================================================================
Web application server providing interactive REST endpoints for historical
crime intelligence analytics and predictive forecasting.
"""
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from analytics import (
    CATEGORIES,
    DATAFRAME,
    category_breakdown,
    forecast,
    series_for,
    summary,
)

# Initialize Flask application
app = Flask(__name__)


# =============================================================================
# View Routes
# =============================================================================
# India district GeoJSON file path
BASE_DIR = Path(__file__).resolve().parent

GEOJSON_PATH = (
    BASE_DIR / "data" / "india_districts.geojson"
)

print("GeoJSON file:", GEOJSON_PATH)
print("GeoJSON exists:", GEOJSON_PATH.exists())


def normalize_map_name(value):
    return str(value).strip().upper()


@app.get("/")
def home():
    """Render the primary dashboard single-page interface."""
    return render_template("index.html")


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.get("/api/options")
def options():
    """
    Return available filter options: states, districts for the selected state,
    available years, and crime categories.
    """
    selected_state = request.args.get("state", "All India").strip()

    if selected_state != "All India":
        matched = DATAFRAME[DATAFRAME.state == selected_state.upper()]
        districts = sorted(matched.district.unique().tolist())
    else:
        districts = []

    return jsonify({
        "states": ["All India"] + sorted(DATAFRAME.state.unique().tolist()),
        "districts": ["All districts"] + districts,
        "years": sorted(DATAFRAME.year.unique().tolist()),
        "categories": CATEGORIES,
    })


@app.get("/api/summary")
def api_summary():
    """
    Return high-level key performance metrics for the selected state and district.
    """
    state = request.args.get("state", "All India")
    district = request.args.get("district", "All districts")

    data = summary(state=state, district=district)
    return jsonify(data)


@app.get("/api/trends")
def trends():
    """
    Return annual time series records with year-over-year change calculations.
    """
    category = request.args.get("category", "total")
    state = request.args.get("state", "All India")
    district = request.args.get("district", "All districts")

    series = series_for(category=category, state=state, district=district)
    return jsonify({"series": series})


@app.get("/api/breakdown")
def breakdown():
    """
    Return crime category distribution for the selected geography and year.
    """
    state = request.args.get("state", "All India")
    district = request.args.get("district", "All districts")
    year = request.args.get("year")

    items = category_breakdown(state=state, district=district, year=year)
    return jsonify({"items": items})


@app.get("/api/forecast")
def api_forecast():
    """
    Generate future crime trend projections using historical linear modeling.
    """
    category = request.args.get("category", "total")
    state = request.args.get("state", "All India")
    district = request.args.get("district", "All districts")
    horizon = request.args.get("horizon", 5)

    result = forecast(
        category=category,
        state=state,
        district=district,
        horizon=horizon,
    )
    status_code = 400 if "error" in result else 200
    return jsonify(result), status_code


@app.get("/api/places")
def places():
    """
    Return top 10 locations (states or districts) ranked by recorded crime count.
    """
    state = request.args.get("state", "All India").strip()
    category = request.args.get("category", "total").strip()
    year = request.args.get("year")

    filtered_df = DATAFRAME.copy()

    if state != "All India":
        filtered_df = filtered_df[filtered_df.state == state.upper()]

    if year and str(year).lower() != "all":
        try:
            filtered_df = filtered_df[filtered_df.year == int(year)]
        except ValueError:
            pass

    group_col = "state" if state == "All India" else "district"
    metric_col = category if category in CATEGORIES else "total"

    top_rows = (
        filtered_df.groupby(group_col)[metric_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    items = [
        {"name": str(location), "count": int(count)}
        for location, count in top_rows.items()
    ]
    return jsonify({"items": items})


# =============================================================================
# Main Application Launcher
# =============================================================================

@app.get("/api/crime-map")
def api_crime_map():

    if not GEOJSON_PATH.exists():
        return jsonify({"error": "India GeoJSON file not found"}), 404

    state = request.args.get("state", "All India")
    category = request.args.get("category", "total")
    year = request.args.get("year", "2014")

    if category not in CATEGORIES:
        category = "total"

    try:
        year = int(year)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid year"}), 400

    # Filter crime records
    df = DATAFRAME[DATAFRAME["year"] == year].copy()

    if state != "All India":
        df = df[df["state"] == state.upper()]

    # Aggregate crimes by state and district
    grouped = (
        df.groupby(["state", "district"])[category]
        .sum()
        .to_dict()
    )

    # Load geographical boundaries
    with open(GEOJSON_PATH, "r", encoding="utf-8") as file:
        geo_data = json.load(file)

    output_features = []

    for feature in geo_data["features"]:

        props = feature["properties"]

        # Get DataMeet district and state names
        district = normalize_map_name(
            props.get("DISTRICT", "")
        )

        state_name = normalize_map_name(
            props.get("ST_NM", "")
        )

        crime_count = grouped.get(
            (state_name, district)
        )

        # Keep the selected state or all India
        if state != "All India" and state_name != state.upper():
            continue

        if crime_count is None:
            crime_count = 0

        feature["properties"]["district"] = district
        feature["properties"]["state"] = state_name
        feature["properties"]["crime_count"] = int(crime_count)
        feature["properties"]["year"] = year
        feature["properties"]["category"] = category

        output_features.append(feature)

    return jsonify({
        "type": "FeatureCollection",
        "features": output_features,
        "year": year,
        "category": category
    })
if __name__ == "__main__":
    print("\nREGISTERED FLASK ROUTES:")

    for route in app.url_map.iter_rules():
        print(route)

    app.run(debug=True)
    record_count = len(DATAFRAME)
    min_year = DATAFRAME.year.min()
    max_year = DATAFRAME.year.max()
    print(f"Loaded {record_count:,} district-year records ({min_year}–{max_year}).")
    app.run(host="127.0.0.1", port=5000, debug=True)
