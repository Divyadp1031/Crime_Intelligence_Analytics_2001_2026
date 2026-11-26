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
 
@app.route("/api/women-safety")
def women_safety():
    try:
        df = DATAFRAME.copy()

        selected_year = request.args.get("year", "all").lower()

        # Filter by selected year when a specific year is requested
        if selected_year != "all":
            year = int(selected_year)
            df = df[df["year"] == year]

            if df.empty:
                return jsonify({
                    "error": "No data available for this year"
                }), 404
        else:
            year = "all"

        # Total recorded cases
        total_cases = int(df["rape"].sum())

        # State-wise analysis
        state_data = (
            df.groupby("state")["rape"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        # District-wise analysis: Top 10
        district_data = (
            df.groupby(["state", "district"])["rape"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        # Year-wise trend
        yearly_data = (
            df.groupby("year")["rape"]
            .sum()
            .sort_index()
            .reset_index()
        )

        return jsonify({
            "selected_year": year,
            "total_cases": total_cases,
            "state_analysis": state_data.to_dict(orient="records"),
            "district_analysis": district_data.to_dict(orient="records"),
            "yearly_analysis": yearly_data.to_dict(orient="records")
        })

    except ValueError:
        return jsonify({"error": "Invalid year"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

PREVENTION_CATEGORIES = [
    ("murder", "Murder"),
    ("rape", "Rape"),
    ("theft", "Theft"),
    ("robbery", "Robbery"),
    ("riots", "Riots"),
]

STABILITY_THRESHOLD_PCT = 5.0

PREVENTION_GUIDANCE = {
    "murder": {
        "increasing": (
            "Consider strengthening community policing, local conflict-resolution "
            "initiatives, and coordination with local authorities."
        ),
        "newly_recorded": (
            "Review newly recorded incidents with local authorities and support "
            "early conflict-resolution and community policing efforts."
        ),
        "stable": (
            "Maintain existing community policing coordination and periodic review "
            "of dispute-prone areas."
        ),
        "decreasing": (
            "Continue current community engagement and inter-agency coordination "
            "while monitoring local indicators."
        ),
    },
    "rape": {
        "increasing": (
            "Review emergency assistance access, street lighting coverage, "
            "safe transit corridors, and survivor support services."
        ),
        "newly_recorded": (
            "Assess emergency response accessibility, public lighting, and "
            "awareness of support services in the area."
        ),
        "stable": (
            "Sustain regular audits of street lighting, helpline responsiveness, "
            "and support service availability."
        ),
        "decreasing": (
            "Maintain women safety infrastructure, emergency access checks, "
            "and community awareness programs."
        ),
    },
    "theft": {
        "increasing": (
            "Promote property-security awareness, improve surveillance coverage, "
            "and review vulnerability in commercial and residential zones."
        ),
        "newly_recorded": (
            "Encourage property-security precautions and assess CCTV or lighting "
            "coverage around affected neighborhoods."
        ),
        "stable": (
            "Continue public property-security awareness and routine maintenance "
            "of local surveillance systems."
        ),
        "decreasing": (
            "Sustain neighborhood watch outreach and periodic checks on "
            "property-security measures."
        ),
    },
    "robbery": {
        "increasing": (
            "Review security arrangements along transit and commercial corridors "
            "and consider focused patrols during peak hours."
        ),
        "newly_recorded": (
            "Evaluate corridor security arrangements and targeted patrol "
            "coverage in emerging hotspots."
        ),
        "stable": (
            "Maintain routine patrol visibility and review security arrangements "
            "at high-footfall locations."
        ),
        "decreasing": (
            "Keep up focused patrol scheduling and coordination with local "
            "commercial and transit stakeholders."
        ),
    },
    "riots": {
        "increasing": (
            "Strengthen community engagement, local dialogue platforms, and "
            "proactive conflict-mediation initiatives."
        ),
        "newly_recorded": (
            "Initiate community dialogue and local mediation channels to address "
            "emerging public-order tensions early."
        ),
        "stable": (
            "Sustain local peace-committee engagement and preventive "
            "conflict-mediation outreach."
        ),
        "decreasing": (
            "Maintain ongoing community dialogue and collaborative local "
            "mediation mechanisms."
        ),
    },
}


def _classify_data_source(years):
    """Classify a list of years into official, synthetic, or mixed source metadata."""
    has_official = any(year <= 2014 for year in years)
    has_synthetic = any(year >= 2015 for year in years)

    if has_official and has_synthetic:
        return {
            "type": "mixed",
            "label": "Mixed Data (Official NCRB 2001–2014 + Synthetic Demonstration 2015–2026)",
        }
    if has_synthetic:
        return {
            "type": "synthetic",
            "label": "Synthetic Demonstration Data (2015–2026)",
        }
    return {
        "type": "official",
        "label": "Official NCRB Historical Data (2001–2014)",
    }


def _summarize_period(geo_df, start_year, end_year, category_keys):
    """Aggregate yearly totals and track observed vs missing years in a period."""
    expected_years = list(range(start_year, end_year + 1))
    period_df = geo_df[geo_df["year"].between(start_year, end_year)]

    if period_df.empty:
        return {
            "start_year": start_year,
            "end_year": end_year,
            "expected_year_count": len(expected_years),
            "observed_years": [],
            "missing_years": expected_years,
            "yearly_averages": {key: None for key in category_keys},
            "source": _classify_data_source(expected_years),
        }

    yearly_totals = period_df.groupby("year")[category_keys].sum().sort_index()
    observed_years = [int(y) for y in yearly_totals.index.tolist()]
    missing_years = [y for y in expected_years if y not in observed_years]
    yearly_averages = {
        key: float(yearly_totals[key].mean())
        for key in category_keys
    }

    return {
        "start_year": start_year,
        "end_year": end_year,
        "expected_year_count": len(expected_years),
        "observed_years": observed_years,
        "missing_years": missing_years,
        "yearly_averages": yearly_averages,
        "source": _classify_data_source(expected_years),
    }


def _classify_trend(prev_avg, recent_avg):
    """Classify trend status and compute percentage change safely."""
    if prev_avg is None or recent_avg is None:
        return "Insufficient Data", None

    if prev_avg == 0 and recent_avg == 0:
        return "No Recorded Cases", None

    if prev_avg == 0 and recent_avg > 0:
        return "Newly Recorded", None

    pct_change = ((recent_avg - prev_avg) / prev_avg) * 100.0
    rounded_pct = round(pct_change, 2)

    if pct_change > STABILITY_THRESHOLD_PCT:
        return "Increased", rounded_pct
    if pct_change < -STABILITY_THRESHOLD_PCT:
        return "Decreased", rounded_pct
    return "Relatively Stable", rounded_pct


@app.get("/api/prevention-analysis")
def api_prevention_analysis():
    """
    Compare crime category averages between two non-overlapping periods for a
    selected geography and return trend classifications and prevention guidance.
    """
    min_year = int(DATAFRAME["year"].min())
    max_year = int(DATAFRAME["year"].max())

    try:
        prev_start = int(request.args.get("prev_start", 2019))
        prev_end = int(request.args.get("prev_end", 2021))
        recent_start = int(request.args.get("recent_start", 2022))
        recent_end = int(request.args.get("recent_end", 2024))
    except (ValueError, TypeError):
        return jsonify({"error": "All period years must be valid integers."}), 400

    for yr in (prev_start, prev_end, recent_start, recent_end):
        if yr < min_year or yr > max_year:
            return jsonify({
                "error": f"Selected years must be within the available range ({min_year}–{max_year})."
            }), 400

    if prev_start > prev_end:
        return jsonify({
            "error": "Previous period start year cannot be after its end year."
        }), 400

    if recent_start > recent_end:
        return jsonify({
            "error": "Recent period start year cannot be after its end year."
        }), 400

    if prev_end >= recent_start:
        return jsonify({
            "error": (
                "Selected periods must not overlap, and the previous period "
                "must end before the recent period starts."
            )
        }), 400

    prev_span = prev_end - prev_start + 1
    recent_span = recent_end - recent_start + 1
    if prev_span != recent_span:
        return jsonify({
            "error": (
                f"Both periods must have the same number of years "
                f"(previous has {prev_span} year{'s' if prev_span != 1 else ''}, "
                f"recent has {recent_span} year{'s' if recent_span != 1 else ''})."
            )
        }), 400

    state = request.args.get("state", "All India").strip() or "All India"
    district = request.args.get("district", "All districts").strip() or "All districts"

    geo_df = DATAFRAME
    if state != "All India":
        geo_df = geo_df[geo_df["state"] == state.upper()]
        if geo_df.empty:
            return jsonify({"error": f"No records found for state '{state}'."}), 404
        resolved_state = state.upper()
    else:
        resolved_state = "All India"

    if district != "All districts":
        if resolved_state == "All India":
            return jsonify({
                "error": "Please select a specific state before selecting a district."
            }), 400
        geo_df = geo_df[geo_df["district"] == district.upper()]
        if geo_df.empty:
            return jsonify({
                "error": f"No records found for district '{district}' in {resolved_state}."
            }), 404
        resolved_district = district.upper()
    else:
        resolved_district = "All districts"

    category_keys = [key for key, _ in PREVENTION_CATEGORIES]
    prev_summary = _summarize_period(geo_df, prev_start, prev_end, category_keys)
    recent_summary = _summarize_period(geo_df, recent_start, recent_end, category_keys)

    categories_result = []
    recommendations = []

    for key, label in PREVENTION_CATEGORIES:
        prev_avg = prev_summary["yearly_averages"][key]
        recent_avg = recent_summary["yearly_averages"][key]
        trend_status, pct_change = _classify_trend(prev_avg, recent_avg)

        categories_result.append({
            "key": key,
            "category": label,
            "previous_avg": round(prev_avg, 2) if prev_avg is not None else None,
            "recent_avg": round(recent_avg, 2) if recent_avg is not None else None,
            "percentage_change": pct_change,
            "trend_status": trend_status,
        })

        if trend_status == "Increased":
            recommendations.append({
                "category": label,
                "trend_status": trend_status,
                "priority": "Focus",
                "suggestion": PREVENTION_GUIDANCE[key]["increasing"],
            })
        elif trend_status == "Newly Recorded":
            recommendations.append({
                "category": label,
                "trend_status": trend_status,
                "priority": "Monitor",
                "suggestion": PREVENTION_GUIDANCE[key]["newly_recorded"],
            })

    if not recommendations:
        has_any_data = any(
            item["trend_status"] != "Insufficient Data"
            for item in categories_result
        )
        if has_any_data:
            recommendations.append({
                "category": "All Tracked Categories",
                "trend_status": "Stable / Decreasing",
                "priority": "Maintain",
                "suggestion": (
                    "No tracked crime categories showed an increase above the "
                    "stability threshold. Continue existing preventive initiatives, "
                    "community partnerships, and routine monitoring."
                ),
            })
        else:
            recommendations.append({
                "category": "Data Coverage",
                "trend_status": "Insufficient Data",
                "priority": "Review",
                "suggestion": (
                    "One or both selected periods have no recorded entries for this "
                    "geography. Select a broader year range or state-level view."
                ),
            })

    all_selected_years = (
        list(range(prev_start, prev_end + 1))
        + list(range(recent_start, recent_end + 1))
    )
    overall_source = _classify_data_source(all_selected_years)

    return jsonify({
        "filters": {
            "state": resolved_state,
            "district": resolved_district,
            "prev_start": prev_start,
            "prev_end": prev_end,
            "recent_start": recent_start,
            "recent_end": recent_end,
        },
        "stability_threshold_pct": STABILITY_THRESHOLD_PCT,
        "data_source": overall_source,
        "periods": {
            "previous": {
                "start_year": prev_summary["start_year"],
                "end_year": prev_summary["end_year"],
                "expected_year_count": prev_summary["expected_year_count"],
                "observed_year_count": len(prev_summary["observed_years"]),
                "observed_years": prev_summary["observed_years"],
                "missing_years": prev_summary["missing_years"],
                "source": prev_summary["source"],
            },
            "recent": {
                "start_year": recent_summary["start_year"],
                "end_year": recent_summary["end_year"],
                "expected_year_count": recent_summary["expected_year_count"],
                "observed_year_count": len(recent_summary["observed_years"]),
                "observed_years": recent_summary["observed_years"],
                "missing_years": recent_summary["missing_years"],
                "source": recent_summary["source"],
            },
        },
        "categories": categories_result,
        "recommendations": recommendations,
        "disclaimer": (
            "General decision-support suggestions only. These recommendations are "
            "based on period comparisons of recorded/demonstration data and do not "
            "guarantee crime reduction."
        ),
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
