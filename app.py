"""
CrimeLens - Flask Application Entry Point
================================================================================
Web application server providing interactive REST endpoints for historical
crime intelligence analytics and predictive forecasting.
"""

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

if __name__ == "__main__":
    record_count = len(DATAFRAME)
    min_year = DATAFRAME.year.min()
    max_year = DATAFRAME.year.max()
    print(f"Loaded {record_count:,} district-year records ({min_year}–{max_year}).")
    app.run(host="127.0.0.1", port=5000, debug=True)
