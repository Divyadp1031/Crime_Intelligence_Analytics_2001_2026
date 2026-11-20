"""
CrimeLens - Crime Analytics & Forecasting Engine
================================================================================
Core data loading, normalization, aggregation, time series calculations,
and machine learning trend projection engine for Indian crime intelligence data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================================================================
# 1. Configuration & Constants
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Historical NCRB source file specifications (filename, baseline year hint)
FILES = [
    ("01_District_wise_crimes_committed_IPC_2001_2012.csv", 2001),
    ("01_District_wise_crimes_committed_IPC_2013.csv", 2013),
    ("01_District_wise_crimes_committed_IPC_2014.csv", 2014),
]

# Tracked crime categories mapping (key identifier -> human-readable label)
CATEGORIES: Dict[str, str] = {
    "total": "Total IPC Crimes",
    "murder": "Murder",
    "rape": "Rape",
    "theft": "Theft",
    "robbery": "Robbery",
    "riots": "Riots",
}

# Strategic allocation crime weighting profiles
STRATEGIES: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "label": "All Crime Types (General Safety)",
        "description": "Balances police presence across violent crimes, property theft, and street disputes.",
        "weights": {"murder": 10.0, "rape": 8.0, "robbery": 5.0, "riots": 4.0, "theft": 1.5},
    },
    "violent": {
        "label": "Violent Crimes (Murder & Assault)",
        "description": "Sends more officers to areas with homicides, assaults, and serious violence.",
        "weights": {"murder": 15.0, "rape": 12.0, "robbery": 6.0, "riots": 3.0, "theft": 0.5},
    },
    "property": {
        "label": "Theft & Commercial Areas",
        "description": "Focuses police patrols on crowded markets, vehicle theft spots, and robberies.",
        "weights": {"theft": 6.0, "robbery": 7.0, "murder": 2.0, "rape": 2.0, "riots": 1.0},
    },
    "riots": {
        "label": "Riots & Public Clashes",
        "description": "Sends quick-response riot control teams to areas with street clashes and unrest.",
        "weights": {"riots": 15.0, "robbery": 5.0, "murder": 4.0, "rape": 2.0, "theft": 1.0},
    },
}

# Travel safety comparison weighting profiles
TRAVEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "general": {
        "label": "General Travel (Family & Leisure)",
        "description": "Standard safety check balancing violent crime and street theft.",
        "weights": {"murder": 8.0, "rape": 7.0, "robbery": 4.0, "riots": 3.0, "theft": 1.5},
    },
    "women": {
        "label": "Women / Solo Traveler",
        "description": "Places extra priority on personal safety, harassment prevention, and low violent crime.",
        "weights": {"rape": 12.0, "murder": 8.0, "robbery": 4.0, "riots": 2.0, "theft": 1.0},
    },
    "tourist": {
        "label": "Tourist & Sightseeing",
        "description": "Focuses on pickpocketing, street scams, vehicle theft, and market robberies.",
        "weights": {"theft": 6.0, "robbery": 7.0, "riots": 4.0, "murder": 2.0, "rape": 2.0},
    },
}


# =============================================================================
# 2. Internal Helpers
# =============================================================================

def _num(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Search for the first available column name from a list of candidates
    and return it parsed into numeric values with NaN filled with 0.

    Args:
        df: Source pandas DataFrame.
        candidates: Ordered list of candidate column names.

    Returns:
        pd.Series: Cleaned numeric series.
    """
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


# =============================================================================
# 3. Data Ingestion & Preprocessing Pipeline
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    Ingest, normalize, and merge historical NCRB crime statistics (2001–2014)
    with demonstration projections (2015–2026).

    Normalizations applied:
      - Strips and uppercases state and district names.
      - Filters out aggregate 'TOTAL' rows to prevent double-counting.
      - Standardizes schema across heterogeneous reporting periods (2014 format shift).
      - Deduplicates entries and aligns category metrics.

    Returns:
        pd.DataFrame: Consolidated dataset sorted by (year, state, district).
    """
    frames: List[pd.DataFrame] = []

    # 1. Ingest official historical NCRB CSV files
    for filename, year_hint in FILES:
        path = DATA / filename
        if not path.exists():
            continue

        raw = pd.read_csv(path, low_memory=False)

        state_col = "STATE/UT" if "STATE/UT" in raw.columns else "States/UTs"
        district_col = "DISTRICT" if "DISTRICT" in raw.columns else "District"
        year_col = "YEAR" if "YEAR" in raw.columns else "Year"
        total_candidates = ["TOTAL IPC CRIMES", "Total Cognizable IPC crimes", "Count"]

        part = pd.DataFrame({
            "state": raw[state_col].astype(str).str.strip().str.upper(),
            "district": raw[district_col].astype(str).str.strip().str.upper(),
            "year": (
                pd.to_numeric(raw[year_col], errors="coerce")
                .fillna(year_hint)
                .astype(int)
            ),
            "total": _num(raw, total_candidates),
            "murder": _num(raw, ["MURDER", "Murder"]),
            "rape": _num(raw, ["RAPE", "Rape"]),
            "theft": _num(raw, ["THEFT", "Theft"]),
            "robbery": _num(raw, ["ROBBERY", "Robbery"]),
            "riots": _num(raw, ["RIOTS", "Riots"]),
        })

        # Exclude aggregate total rows to prevent double-counting
        part = part[~part["district"].str.contains(r"TOTAL|ZZ TOTAL", na=False)]
        part = part[~part["state"].str.contains("TOTAL", na=False)]
        part = part[(part["state"].str.len() > 1) & (part["district"].str.len() > 1)]
        part = part.drop_duplicates(subset=["state", "district", "year"])

        frames.append(part)

    if not frames:
        raise FileNotFoundError("NCRB CSV files were not found in the data folder.")

    # 2. Ingest synthetic demonstration records for 2015-2026
    synthetic_path = DATA / "SYNTHETIC_demo_crime_data_2015_2026.csv"
    if synthetic_path.exists():
        demo = pd.read_csv(synthetic_path, low_memory=False)
        demo["year"] = pd.to_numeric(demo["year"], errors="coerce")
        demo = demo[demo["year"].between(2015, 2026)].copy()

        if not demo.empty:
            demo["year"] = demo["year"].astype(int)
            demo["state"] = demo["state"].astype(str).str.strip().str.upper()
            demo["district"] = demo["district"].astype(str).str.strip().str.upper()

            for col in CATEGORIES:
                demo[col] = pd.to_numeric(demo[col], errors="coerce").fillna(0.0)

            category_columns = list(CATEGORIES.keys())
            frames.append(demo[["state", "district", "year"] + category_columns])

    # 3. Concatenate and aggregate duplicates
    data = pd.concat(frames, ignore_index=True)
    data = data[data["year"].between(2001, 2026)]
    data = (
        data.groupby(["state", "district", "year"], as_index=False)[list(CATEGORIES.keys())]
        .sum()
    )

    return data.sort_values(["year", "state", "district"])


# Global instantiated dataset
DATAFRAME: pd.DataFrame = load_data()


# =============================================================================
# 4. Query Filtering & Time Series Analysis
# =============================================================================

def _filtered(state: str = "All India", district: str = "All districts") -> pd.DataFrame:
    """Filter the master dataset by state and district."""
    df = DATAFRAME
    if state and state != "All India":
        df = df[df.state == state.upper()]
    if district and district != "All districts":
        df = df[df.district == district.upper()]
    return df


def series_for(
    category: str = "total",
    state: str = "All India",
    district: str = "All districts",
) -> List[Dict[str, Any]]:
    """
    Generate chronological yearly time series with YoY change calculations.

    Args:
        category: Selected crime category key.
        state: Selected State/UT or 'All India'.
        district: Selected District or 'All districts'.

    Returns:
        List of dictionaries containing year, count, change, and change_pct.
    """
    cat_key = category if category in CATEGORIES else "total"
    filtered_df = _filtered(state, district)

    if filtered_df.empty:
        return []

    grouped = filtered_df.groupby("year")[cat_key].sum().sort_index()

    series: List[Dict[str, Any]] = []
    previous_count: Optional[int] = None

    for year, value in grouped.items():
        current_count = int(round(value))
        if previous_count is None:
            pct_change = None
            abs_change = None
        else:
            abs_change = current_count - previous_count
            pct_change = (
                round((current_count - previous_count) / previous_count * 100, 2)
                if previous_count > 0
                else None
            )

        series.append({
            "year": int(year),
            "count": current_count,
            "change_pct": pct_change,
            "change": abs_change,
        })
        previous_count = current_count

    return series


def summary(
    state: str = "All India",
    district: str = "All districts",
) -> Dict[str, Any]:
    """
    Compute aggregate summary statistics for the selected geography.

    Returns:
        Dictionary containing total records, state count, district count,
        total crimes, year range, and top places breakdown.
    """
    df = _filtered(state, district)
    years = sorted(df.year.unique().tolist())
    total_crimes = int(df.total.sum()) if not df.empty else 0

    group_col = "state" if state == "All India" else "district"
    top_series = (
        df.groupby(group_col).total
        .sum()
        .sort_values(ascending=False)
        .head(8)
    )

    return {
        "records": int(len(df)),
        "states": int(df.state.nunique()),
        "districts": int(df.district.nunique()),
        "total_crimes": total_crimes,
        "first_year": int(min(years)) if years else None,
        "last_year": int(max(years)) if years else None,
        "top_places": [
            {"name": str(place), "count": int(count)}
            for place, count in top_series.items()
        ],
    }


def category_breakdown(
    state: str = "All India",
    district: str = "All districts",
    year: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregate counts across individual crime categories (excluding 'total')
    for a given geography and optional specific year.
    """
    df = _filtered(state, district)

    if year not in (None, "all", "All years", ""):
        try:
            df = df[df.year == int(year)]
        except ValueError:
            pass

    return [
        {
            "category": CATEGORIES[cat_key],
            "count": int(round(df[cat_key].sum())),
        }
        for cat_key in CATEGORIES
        if cat_key != "total"
    ]


# =============================================================================
# 5. Machine Learning Forecasting Engine
# =============================================================================

def forecast(
    category: str = "total",
    state: str = "All India",
    district: str = "All districts",
    horizon: int = 5,
) -> Dict[str, Any]:
    """
    Generate linear regression forecasts for future years with chronological
    holdout validation.

    Args:
        category: Crime category key.
        state: Filter state.
        district: Filter district.
        horizon: Number of future years to forecast (e.g., 3 or 5).

    Returns:
        Dictionary containing historical series, future predictions,
        trend direction, holdout MAE and RMSE metrics, and model notes.
    """
    history = series_for(category, state, district)
    if len(history) < 4:
        return {
            "error": "At least four annual observations are required for a meaningful trend forecast."
        }

    try:
        horizon_years = int(horizon)
    except (ValueError, TypeError):
        horizon_years = 5

    years_matrix = np.array([item["year"] for item in history], dtype=float).reshape(-1, 1)
    counts_vector = np.array([item["count"] for item in history], dtype=float)

    # Chronological holdout evaluation:
    # Use final 2 years if >=7 observations, else final 1 year
    test_size = 2 if len(history) >= 7 else 1
    train_x = years_matrix[:-test_size]
    train_y = counts_vector[:-test_size]
    test_x = years_matrix[-test_size:]
    test_y = counts_vector[-test_size:]

    eval_model = LinearRegression()
    eval_model.fit(train_x, train_y)
    test_predictions = np.maximum(0.0, eval_model.predict(test_x))

    mae = float(mean_absolute_error(test_y, test_predictions))
    rmse = float(np.sqrt(mean_squared_error(test_y, test_predictions)))

    # Fit full production model across all historical points
    full_model = LinearRegression()
    full_model.fit(years_matrix, counts_vector)

    last_year = int(years_matrix[-1, 0])
    future_years = np.arange(last_year + 1, last_year + horizon_years + 1).reshape(-1, 1)
    future_predictions = np.maximum(0.0, full_model.predict(future_years))

    forecast_items = [
        {
            "year": int(year),
            "predicted_count": int(round(pred)),
        }
        for year, pred in zip(future_years.ravel(), future_predictions)
    ]

    slope = float(full_model.coef_[0])
    if slope > 0.5:
        direction = "Increasing"
    elif slope < -0.5:
        direction = "Decreasing"
    else:
        direction = "Relatively stable"

    return {
        "history": history,
        "forecast": forecast_items,
        "direction": direction,
        "annual_change": round(slope, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "model": "Linear Regression",
        "note": (
            "Trend-based estimate using annual historical counts. "
            "It is not a guarantee and does not account for population changes, "
            "reporting practices, policy changes or other external factors."
        ),
    }


# =============================================================================
# 6. India Crime Risk Map Engine (High, Medium, Low by Crime Category)
# =============================================================================

MAP_CATEGORIES: Dict[str, str] = {
    "total": "Total Reported IPC Crimes",
    "murder": "Murder & Violent Crimes",
    "rape": "Women Safety (Rape & Assault)",
    "theft": "Theft & Pickpocketing",
    "robbery": "Robbery & Armed Heists",
    "riots": "Riots & Public Clashes",
}

MAP_STATE_CODES: Dict[str, str] = {
    "an": "A & N ISLANDS",
    "ap": "ANDHRA PRADESH",
    "ar": "ARUNACHAL PRADESH",
    "as": "ASSAM",
    "br": "BIHAR",
    "ch": "CHANDIGARH",
    "ct": "CHHATTISGARH",
    "dn": "D & N HAVELI",
    "dd": "DAMAN & DIU",
    "dl": "DELHI UT",
    "ga": "GOA",
    "gj": "GUJARAT",
    "hr": "HARYANA",
    "hp": "HIMACHAL PRADESH",
    "jk": "JAMMU & KASHMIR",
    "jh": "JHARKHAND",
    "ka": "KARNATAKA",
    "kl": "KERALA",
    "ld": "LAKSHADWEEP",
    "mp": "MADHYA PRADESH",
    "mh": "MAHARASHTRA",
    "mn": "MANIPUR",
    "ml": "MEGHALAYA",
    "mz": "MIZORAM",
    "nl": "NAGALAND",
    "or": "ODISHA",
    "py": "PUDUCHERRY",
    "pb": "PUNJAB",
    "rj": "RAJASTHAN",
    "sk": "SIKKIM",
    "tn": "TAMIL NADU",
    "tg": "TELANGANA",
    "tr": "TRIPURA",
    "up": "UTTAR PRADESH",
    "ut": "UTTARAKHAND",
    "wb": "WEST BENGAL",
}


def get_india_risk_map(
    category: str = "total",
    year: int = 2026,
) -> Dict[str, Any]:
    """
    Generate state-by-state risk level classification (High, Medium, Low)
    across all 36 Indian states and union territories for a selected crime category.

    Args:
        category: Selected crime category ('total', 'murder', 'rape', 'theft', 'robbery', 'riots').
        year: Analysis or projection year (e.g. 2026, 2027, 2028).

    Returns:
        Structured payload containing map coordinates, risk colors, counts,
        state rankings, and top dangerous districts.
    """
    cat_key = category if category in CATEGORIES else "total"
    friendly_label = MAP_CATEGORIES.get(cat_key, "Total Reported IPC Crimes")

    try:
        target_year = int(year)
    except (ValueError, TypeError):
        target_year = 2026

    df = DATAFRAME
    max_hist_year = int(df.year.max())
    items: List[Dict[str, Any]] = []

    for code, sname in MAP_STATE_CODES.items():
        sub = df[(df.state == sname) | (df.state == sname.replace(" ", ""))]
        if sub.empty:
            continue

        if target_year <= max_hist_year:
            sub_yr = sub[sub.year == target_year]
            if sub_yr.empty:
                sub_yr = sub[sub.year == max_hist_year]
            val = int(round(sub_yr[cat_key].sum()))

            top_d = sub_yr.groupby("district")[cat_key].sum().sort_values(ascending=False)
            top_dist_name = top_d.index[0] if len(top_d) > 0 else "General"
            top_dist_val = int(round(top_d.iloc[0])) if len(top_d) > 0 else 0
        else:
            # Forecast future year using linear regression
            yearly_series = sub.groupby("year")[cat_key].sum().sort_index()
            X = np.array(yearly_series.index, dtype=float).reshape(-1, 1)
            y = np.array(yearly_series.values, dtype=float)

            if len(X) >= 3:
                model = LinearRegression().fit(X, y)
                pred_val = max(0.0, float(model.predict([[target_year]])[0]))
            else:
                pred_val = max(0.0, float(y.mean()))

            val = int(round(pred_val))

            recent_sub = sub[sub.year >= max_hist_year - 2]
            if recent_sub.empty:
                recent_sub = sub
            top_d = recent_sub.groupby("district")[cat_key].sum().sort_values(ascending=False)
            top_dist_name = top_d.index[0] if len(top_d) > 0 else "General"
            top_dist_val = (
                int(round(top_d.iloc[0] / max(1, len(recent_sub.year.unique()))))
                if len(top_d) > 0
                else 0
            )

        items.append({
            "code": code,
            "state": sname.title(),
            "count": val,
            "top_district": top_dist_name.title(),
            "top_district_count": top_dist_val,
        })

    items.sort(key=lambda x: x["count"], reverse=True)
    total_national_crimes = sum(item["count"] for item in items)

    counts_array = np.array([x["count"] for x in items], dtype=float)
    p67 = float(np.percentile(counts_array, 67)) if len(counts_array) > 0 else 0.0
    p33 = float(np.percentile(counts_array, 33)) if len(counts_array) > 0 else 0.0

    states_map: Dict[str, Dict[str, Any]] = {}

    for rank, it in enumerate(items, 1):
        it["rank"] = rank
        c = it["count"]
        it["share_pct"] = round((c / max(1, total_national_crimes)) * 100, 1)

        if c >= p67 and c > 0:
            it["risk_level"] = "High"
            it["risk_label"] = "🔴 High Risk"
            it["risk_color"] = "#e74c3c"
            it["risk_badge_class"] = "tier-critical"
        elif c >= p33 and c > 0:
            it["risk_level"] = "Medium"
            it["risk_label"] = "🟡 Medium Risk"
            it["risk_color"] = "#f39c12"
            it["risk_badge_class"] = "tier-elevated"
        else:
            it["risk_level"] = "Low"
            it["risk_label"] = "🟢 Low Risk"
            it["risk_color"] = "#27ae60"
            it["risk_badge_class"] = "tier-baseline"

        states_map[it["code"]] = it

    high_count = sum(1 for it in items if it["risk_level"] == "High")
    medium_count = sum(1 for it in items if it["risk_level"] == "Medium")
    low_count = sum(1 for it in items if it["risk_level"] == "Low")

    return {
        "category": cat_key,
        "category_label": friendly_label,
        "year": target_year,
        "total_national_crimes": total_national_crimes,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "states_map": states_map,
        "leaderboard": items,
        "categories_available": MAP_CATEGORIES,
    }


# =============================================================================
# 6b. District Crime Risk & Incident Predictor Engine
# =============================================================================

INCIDENT_ADVICE: Dict[str, Dict[str, Any]] = {
    "theft": {
        "title": "Stolen Property & Pickpocketing Guidance",
        "action_steps": [
            "🚨 Immediate Action: File an e-FIR immediately via your State Police portal or visit the nearest police station.",
            "📱 Mobile Tracking: Report and block stolen phones on the Government CEIR portal (ceir.gov.in) to prevent unauthorized SIM activation.",
            "💳 Financial Safeguards: Immediately freeze credit/debit cards, UPI apps, and change account passwords if digital devices or wallets are lost.",
            "📍 Local Hotspot Caution: In this district, thefts occur predominantly in busy transit hubs, railway stations, bus stands, and commercial bazaars.",
        ],
        "emergency_number": "112 (National Emergency Helpline)",
    },
    "rape": {
        "title": "Women Safety & Harassment Prevention Guidance",
        "action_steps": [
            "🚨 Emergency Police Response: Dial 112 immediately or call the dedicated Women Helpline at 1091.",
            "🚗 Safe Transit: Use verified rideshare or public transit with live ride-sharing enabled to trusted family or friends.",
            "💡 Route Navigation: Stick to well-lit, CCTV-monitored main avenues; avoid deserted bypass roads and isolated industrial tracts.",
            "🛡️ Support Networks: State One-Stop Centers (OSC) provide 24/7 medical, legal, and counseling aid for women.",
        ],
        "emergency_number": "1091 (Women Helpline) / 112",
    },
    "murder": {
        "title": "Violent Crime & Severe Conflict Guidance",
        "action_steps": [
            "🚨 Urgent Police Intervention: Dial 112 without delay if observing or facing violent threats or armed disputes.",
            "🌙 Nighttime Safety: Avoid secluded peripheral stretches, unpatrolled outskirts, and dimly lit service lanes late at night.",
            "⚠️ Dispute De-escalation: Refrain from engaging in escalated road rage or street altercations; seek safe public premises and alert police.",
        ],
        "emergency_number": "112 (Police & Emergency Dispatch)",
    },
    "robbery": {
        "title": "Mugging & Armed Robbery Guidance",
        "action_steps": [
            "⚠️ Precautionary Awareness: Avoid displaying expensive jewelry, high-end electronics, or handling physical cash in uncrowded areas.",
            "🏧 Safe Cash Withdrawals: Prefer bank-attached ATMs with active security guards over secluded street kiosks.",
            "🚨 Incident Survival: If confronted, prioritize personal safety over possessions; note suspect vehicle/appearance and dial 112 immediately.",
        ],
        "emergency_number": "112 (Police Dispatch)",
    },
    "riots": {
        "title": "Public Unrest & Riot Guidance",
        "action_steps": [
            "📢 Advisory Monitoring: Follow official district administration notices and local news for road blockages or Section 144 restrictions.",
            "🚫 Avoid Hotspots: Steer well clear of volatile political rallies, contentious dispute grounds, and large mob demonstrations.",
            "🏢 Safe Refuge: In case of unexpected street clashes, take shelter inside major commercial establishments or metro facilities.",
        ],
        "emergency_number": "112 (Emergency Services)",
    },
    "total": {
        "title": "General District Safety & Crime Vigilance",
        "action_steps": [
            "📞 Emergency Preparedness: Keep local police control room numbers and 112 saved in speed dial.",
            "🚓 Local Patrols: Take note of localized Police Control Room (PCR) patrol vans and active beat checkpoints in this district.",
            "🛡️ Civic Alert: Report suspicious or unattended packages, vehicle tampering, or disorderly conduct to local authorities.",
        ],
        "emergency_number": "112 (National Emergency Support)",
    },
}


def get_district_risk_analysis(
    state: str = "TAMIL NADU",
    district: str = "CHENNAI",
    category: str = "theft",
    year: int = 2026,
) -> Dict[str, Any]:
    """
    Generate district-level crime risk prediction, most common crime identification,
    and tailored incident advice (e.g. for stolen items, women safety, etc.).

    Args:
        state: State or Union Territory name.
        district: District name.
        category: Selected crime category or incident of interest.
        year: Analysis or forecast year.

    Returns:
        Structured payload containing district metrics, category risk matrix,
        primary crime driver, incident response steps, and state district rankings.
    """
    state_clean = (state or "").strip().upper()
    district_clean = (district or "").strip().upper()
    try:
        target_year = int(year)
    except (ValueError, TypeError):
        target_year = 2026

    cat_key = category if category in CATEGORIES else "theft"

    df = DATAFRAME
    max_hist_year = int(df.year.max())

    # State-level filtering
    sub_state = df[(df.state == state_clean) | (df.state == state_clean.replace(" ", ""))]
    if sub_state.empty:
        # Fallback to first state if not matched
        state_clean = df.state.iloc[0]
        sub_state = df[df.state == state_clean]

    all_districts = sorted(sub_state.district.unique().tolist())
    if district_clean not in all_districts:
        top_d = sub_state.groupby("district")["total"].sum().sort_values(ascending=False)
        district_clean = top_d.index[0] if len(top_d) > 0 else all_districts[0]

    sub_dist = sub_state[sub_state.district == district_clean]

    # Calculate target district counts
    district_counts: Dict[str, int] = {}
    categories_list = ["total", "murder", "rape", "theft", "robbery", "riots"]

    if target_year <= max_hist_year:
        row_yr = sub_dist[sub_dist.year == target_year]
        if row_yr.empty:
            row_yr = sub_dist[sub_dist.year == max_hist_year]
        for c in categories_list:
            district_counts[c] = int(round(float(row_yr[c].sum()))) if not row_yr.empty else 0
    else:
        for c in categories_list:
            yearly = sub_dist.groupby("year")[c].sum().sort_index()
            X = np.array(yearly.index, dtype=float).reshape(-1, 1)
            y = np.array(yearly.values, dtype=float)
            if len(X) >= 3:
                reg = LinearRegression().fit(X, y)
                pred = max(0.0, float(reg.predict([[target_year]])[0]))
            else:
                pred = max(0.0, float(y.mean()))
            district_counts[c] = int(round(pred))

    # Calculate state distribution for all districts
    state_dist_counts: Dict[str, pd.Series] = {}
    for c in categories_list:
        if target_year <= max_hist_year:
            sub_yr = sub_state[sub_state.year == target_year]
            if sub_yr.empty:
                sub_yr = sub_state[sub_state.year == max_hist_year]
            state_dist_counts[c] = sub_yr.groupby("district")[c].sum()
        else:
            recent = sub_state[sub_state.year >= max_hist_year - 2]
            if recent.empty:
                recent = sub_state
            years_count = max(1, len(recent.year.unique()))
            state_dist_counts[c] = recent.groupby("district")[c].sum() / years_count

    # Build category risk matrix for this district
    cat_matrix: Dict[str, Dict[str, Any]] = {}
    for c in categories_list:
        vals = state_dist_counts[c]
        d_val = district_counts[c]
        vals_array = vals.values.astype(float) if len(vals) > 0 else np.array([0.0])
        p67 = float(np.percentile(vals_array, 67)) if len(vals_array) > 0 else 0.0
        p33 = float(np.percentile(vals_array, 33)) if len(vals_array) > 0 else 0.0
        rank = 1 + int((vals > d_val).sum())
        state_avg = float(vals.mean()) if len(vals) > 0 else 0.0

        if d_val >= p67 and d_val > 0:
            risk_level = "High"
            risk_label = "🔴 High Risk"
            risk_color = "#e74c3c"
            risk_badge_class = "tier-critical"
        elif d_val >= p33 and d_val > 0:
            risk_level = "Medium"
            risk_label = "🟡 Medium Risk"
            risk_color = "#f39c12"
            risk_badge_class = "tier-elevated"
        else:
            risk_level = "Low"
            risk_label = "🟢 Low Risk"
            risk_color = "#27ae60"
            risk_badge_class = "tier-baseline"

        # Multi-year trend
        recent_5 = sub_dist.groupby("year")[c].sum().sort_index().tail(5)
        if len(recent_5) >= 2:
            pct_change = ((recent_5.iloc[-1] - recent_5.iloc[0]) / max(1.0, float(recent_5.iloc[0]))) * 100.0
            if pct_change > 5.0:
                trend = "↗️ Increasing"
            elif pct_change < -5.0:
                trend = "↘️ Decreasing"
            else:
                trend = "➡️ Stable"
        else:
            trend = "➡️ Stable"

        diff_pct = round(((d_val - state_avg) / max(1.0, state_avg)) * 100, 1)
        if diff_pct > 0:
            vs_state_avg = f"+{diff_pct}% above state avg"
        elif diff_pct < 0:
            vs_state_avg = f"{abs(diff_pct)}% below state avg"
        else:
            vs_state_avg = "Equal to state avg"

        tot = max(1, district_counts["total"])
        share_of_district = round((d_val / tot) * 100, 1) if c != "total" else 100.0

        cat_matrix[c] = {
            "category": c,
            "category_label": MAP_CATEGORIES.get(c, c.title()),
            "count": d_val,
            "share_of_district": share_of_district,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "risk_badge_class": risk_badge_class,
            "rank": rank,
            "total_districts": len(all_districts),
            "state_avg": int(round(state_avg)),
            "vs_state_avg": vs_state_avg,
            "trend": trend,
        }

    # Identify #1 most common crime category
    indiv_cats = ["theft", "murder", "rape", "robbery", "riots"]
    primary_cat = max(indiv_cats, key=lambda k: district_counts[k])
    primary_count = district_counts[primary_cat]
    primary_share = cat_matrix[primary_cat]["share_of_district"]

    # Incident-specific advice
    advice_info = INCIDENT_ADVICE.get(cat_key, INCIDENT_ADVICE["theft"])

    # Top 5 districts in state for the selected category
    top_in_state = []
    sorted_d = state_dist_counts[cat_key].sort_values(ascending=False).head(5)
    for r, (d_name, d_cnt) in enumerate(sorted_d.items(), 1):
        top_in_state.append({
            "rank": r,
            "district": d_name.title(),
            "count": int(round(d_cnt)),
            "is_current": (d_name == district_clean),
        })

    return {
        "state": state_clean.title(),
        "district": district_clean.title(),
        "year": target_year,
        "selected_category": cat_key,
        "selected_category_label": MAP_CATEGORIES.get(cat_key, "Total Reported IPC Crimes"),
        "total_crimes": district_counts["total"],
        "primary_crime": {
            "category": primary_cat,
            "label": MAP_CATEGORIES.get(primary_cat, primary_cat.title()),
            "count": primary_count,
            "share_pct": primary_share,
        },
        "incident_assessment": {
            "category": cat_key,
            "category_label": MAP_CATEGORIES.get(cat_key, "Total Reported IPC Crimes"),
            "count": district_counts[cat_key],
            "risk_level": cat_matrix[cat_key]["risk_level"],
            "risk_label": cat_matrix[cat_key]["risk_label"],
            "risk_color": cat_matrix[cat_key]["risk_color"],
            "risk_badge_class": cat_matrix[cat_key]["risk_badge_class"],
            "rank": cat_matrix[cat_key]["rank"],
            "total_districts": len(all_districts),
            "trend": cat_matrix[cat_key]["trend"],
            "vs_state_avg": cat_matrix[cat_key]["vs_state_avg"],
            "advice_title": advice_info["title"],
            "action_steps": advice_info["action_steps"],
            "emergency_number": advice_info["emergency_number"],
        },
        "category_matrix": cat_matrix,
        "top_districts_in_state": top_in_state,
        "all_districts_in_state": [d.title() for d in all_districts],
    }


# =============================================================================
# 6c. Age Category & Victim Vulnerability Analysis Engine
# =============================================================================

AGE_GROUPS_CONFIG: List[Dict[str, Any]] = [
    {
        "key": "under_18",
        "label": "Children & Minors (Under 18)",
        "short_label": "Children (<18)",
        "icon": "🧒",
        "color": "#3498db",
        "primary_threat": "School transit safety, kidnapping, and cyber exploitation",
        "advice": "Use verified school transport, save Childline 1098, monitor online screen activity.",
    },
    {
        "key": "18_30",
        "label": "Young Adults (18 to 30)",
        "short_label": "Young Adults (18–30)",
        "icon": "🧑",
        "color": "#e74c3c",
        "primary_threat": "Street theft, smartphone snatching, transit corridor crimes, and cyber fraud",
        "advice": "Keep valuables zipped in bags, avoid using phones near train/bus doors, verify UPI payment QR codes.",
    },
    {
        "key": "30_50",
        "label": "Working Adults (30 to 50)",
        "short_label": "Working Adults (30–50)",
        "icon": "👨‍💼",
        "color": "#3659d9",
        "primary_threat": "Vehicle theft, residential burglary, commercial extortion, and robbery",
        "advice": "Equip vehicles with GPS immobilizers, secure home entrances, avoid leaving valuables in parked cars.",
    },
    {
        "key": "above_50",
        "label": "Senior Citizens (Above 50)",
        "short_label": "Senior Citizens (50+)",
        "icon": "👵",
        "color": "#f39c12",
        "primary_threat": "Chain snatching, doorstep deception scams, and unattended home break-ins",
        "advice": "Save Senior Citizen Helpline 14567 and 112, verify unknown visitors before opening doors.",
    },
]

CRIME_AGE_PROFILES: Dict[str, Dict[str, float]] = {
    "theft": {"under_18": 0.06, "18_30": 0.49, "30_50": 0.33, "above_50": 0.12},
    "rape": {"under_18": 0.254, "18_30": 0.647, "30_50": 0.097, "above_50": 0.002},
    "murder": {"under_18": 0.057, "18_30": 0.357, "30_50": 0.423, "above_50": 0.163},
    "robbery": {"under_18": 0.04, "18_30": 0.46, "30_50": 0.36, "above_50": 0.14},
    "riots": {"under_18": 0.04, "18_30": 0.54, "30_50": 0.34, "above_50": 0.08},
    "total": {"under_18": 0.07, "18_30": 0.48, "30_50": 0.335, "above_50": 0.115},
}


def get_age_group_analysis(
    state: str = "TAMIL NADU",
    district: str = "CHENNAI",
    category: str = "total",
    year: int = 2026,
) -> Dict[str, Any]:
    """
    Calculate crime victim distribution and vulnerability across 4 standard age groups
    (Children, Young Adults, Working Adults, Senior Citizens) for any selected district.

    Args:
        state: State or UT name.
        district: District name.
        category: Selected crime category ('total', 'theft', 'murder', 'rape', 'robbery', 'riots').
        year: Analysis or forecast year.

    Returns:
        Structured payload containing age breakdown counts, percentage shares,
        most affected group highlight, tailored advice, and Chart.js datasets.
    """
    cat_key = category if category in CATEGORIES else "total"
    dist_analysis = get_district_risk_analysis(
        state=state, district=district, category=cat_key, year=year
    )

    state_name = dist_analysis["state"]
    district_name = dist_analysis["district"]
    target_year = dist_analysis["year"]
    category_label = dist_analysis["selected_category_label"]

    base_count = dist_analysis["incident_assessment"]["count"]
    if base_count <= 0:
        base_count = dist_analysis["total_crimes"]

    profile = CRIME_AGE_PROFILES.get(cat_key, CRIME_AGE_PROFILES["total"])

    groups: List[Dict[str, Any]] = []
    total_victims = max(1, base_count)

    for cfg in AGE_GROUPS_CONFIG:
        k = cfg["key"]
        pct = profile.get(k, 0.25)
        raw_val = int(round(base_count * pct))
        share = round((raw_val / total_victims) * 100, 1)

        if share >= 40.0:
            risk_lvl = "Highest Impact"
            badge = "tier-critical"
        elif share >= 25.0:
            risk_lvl = "High Impact"
            badge = "tier-high"
        elif share >= 12.0:
            risk_lvl = "Moderate Impact"
            badge = "tier-elevated"
        else:
            risk_lvl = "Baseline Impact"
            badge = "tier-baseline"

        groups.append({
            "key": k,
            "label": cfg["label"],
            "short_label": cfg["short_label"],
            "icon": cfg["icon"],
            "count": raw_val,
            "share_pct": share,
            "risk_level": risk_lvl,
            "badge_class": badge,
            "color": cfg["color"],
            "primary_threat": cfg["primary_threat"],
            "advice": cfg["advice"],
        })

    # Identify the age group that suffers the most
    most_affected = max(groups, key=lambda x: x["count"])

    return {
        "state": state_name,
        "district": district_name,
        "year": target_year,
        "category": cat_key,
        "category_label": category_label,
        "total_victims": base_count,
        "most_affected": {
            "key": most_affected["key"],
            "label": most_affected["label"],
            "icon": most_affected["icon"],
            "count": most_affected["count"],
            "share_pct": most_affected["share_pct"],
            "summary": (
                f"In {district_name} ({state_name}), {most_affected['label']} suffer the most, "
                f"representing {most_affected['share_pct']}% of all recorded victims "
                f"({most_affected['count']:,} estimated victims/year)."
            ),
            "primary_threat": most_affected["primary_threat"],
            "advice": most_affected["advice"],
        },
        "age_groups": groups,
        "chart_data": {
            "labels": [g["short_label"] for g in groups],
            "values": [g["count"] for g in groups],
            "shares": [g["share_pct"] for g in groups],
            "colors": [g["color"] for g in groups],
        },
    }


# =============================================================================
# 7. Safe Travel & Area Comparison Engine
# =============================================================================

def _generate_travel_tips(breakdown: Dict[str, float], score: int) -> List[str]:
    """Generate friendly, practical travel advice for an area based on crime counts."""
    tips: List[str] = []

    # 1. Theft guidance
    if breakdown.get("theft", 0) > 400:
        tips.append("🎒 Keep wallets, phones, and bags secure in busy markets and train stations.")
    else:
        tips.append("🎒 Petty theft is relatively low. Standard luggage care is sufficient.")

    # 2. Night / Violent crime guidance
    violent = breakdown.get("murder", 0) + breakdown.get("robbery", 0)
    if violent > 100:
        tips.append("🌙 Stick to well-lit main streets and use verified cabs when traveling late at night.")
    else:
        tips.append("🌙 Violent incidents are rare. Generally safe for evening family outings.")

    # 3. Women safety guidance
    if breakdown.get("rape", 0) > 50:
        tips.append("🚕 For solo or women travelers, prefer registered taxis and avoid isolated routes after dark.")
    else:
        tips.append("👩 Women safety indicators are positive for daytime and transit travel.")

    # 4. Overall recommendation based on score
    if score >= 70:
        tips.append("✅ Highly recommended for peaceful travel and family visits.")
    elif score < 45:
        tips.append("⚠️ Stay aware of your surroundings, especially in crowded bus stands or transit hubs.")

    return tips[:3]


def _get_area_travel_stats(
    df: pd.DataFrame,
    state: str,
    district: str,
    weights: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """Calculate normalized safety score, crime breakdown, and travel tips for one area."""
    state_clean = (state or "").strip().upper()
    district_clean = (district or "").strip().upper()

    if district_clean and district_clean != "ALL DISTRICTS":
        sub = df[(df.state == state_clean) & (df.district == district_clean)]
        name = f"{district_clean.title()}, {state_clean.title()}"
    else:
        sub = df[df.state == state_clean]
        name = state_clean.title()

    if sub.empty:
        return None

    years = sorted(sub.year.unique())
    recent_years = years[-3:] if len(years) >= 3 else years
    recent = sub[sub.year.isin(recent_years)]

    cats = ["murder", "rape", "theft", "robbery", "riots"]
    breakdown: Dict[str, float] = {}
    for c in cats:
        breakdown[c] = round(float(recent.groupby("year")[c].sum().mean()), 1)

    total_crimes = int(round(float(recent.groupby("year")["total"].sum().mean())))
    hazard = sum(breakdown[c] * weights.get(c, 1.0) for c in cats)

    # Normalized 0-100 Safety Score (higher is safer)
    score = max(20, min(98, int(round(100 - (np.log1p(hazard) / np.log1p(35000.0)) * 68))))

    if score >= 70:
        label = "🟢 Very Safe"
        desc = "Low reported crime. Ideal for general travel and family visits."
    elif score >= 50:
        label = "🟡 Moderate Safety"
        desc = "Normal city safety. Exercise basic everyday awareness."
    else:
        label = "🔴 Exercise Caution"
        desc = "Higher crime frequency. Stay alert in busy and unfamiliar areas."

    # Trend calculation
    y_sums = sub.groupby("year")["total"].sum()
    if len(y_sums) >= 2:
        y_vals = y_sums.values[-3:]
        diff = (y_vals[-1] - y_vals[0]) / max(1.0, y_vals[0]) * 100.0
        if diff < -3.0:
            trend = "Decreasing (Getting Safer)"
            trend_icon = "📉"
        elif diff > 3.0:
            trend = "Increasing (Rising Crime)"
            trend_icon = "📈"
        else:
            trend = "Steady / Stable"
            trend_icon = "➡️"
    else:
        trend = "Stable"
        trend_icon = "➡️"

    tips = _generate_travel_tips(breakdown, score)

    return {
        "name": name,
        "state": state_clean,
        "district": district_clean,
        "safety_score": score,
        "safety_label": label,
        "safety_desc": desc,
        "hazard_score": round(hazard, 1),
        "total_annual_crimes": total_crimes,
        "trend": trend,
        "trend_icon": trend_icon,
        "crime_breakdown": breakdown,
        "travel_tips": tips,
    }


def compare_travel_safety(
    state_a: str = "TAMIL NADU",
    district_a: str = "CHENNAI",
    state_b: str = "TAMIL NADU",
    district_b: str = "COIMBATORE",
    profile_key: str = "general",
) -> Dict[str, Any]:
    """
    Compare travel safety metrics between two areas and determine which is safer.
    """
    profile = TRAVEL_PROFILES.get(profile_key, TRAVEL_PROFILES["general"])
    weights = profile["weights"]

    area_a = _get_area_travel_stats(DATAFRAME, state_a, district_a, weights)
    area_b = _get_area_travel_stats(DATAFRAME, state_b, district_b, weights)

    if not area_a:
        return {"error": f"No crime records found for Area 1: {district_a}, {state_a}"}
    if not area_b:
        return {"error": f"No crime records found for Area 2: {district_b}, {state_b}"}

    score_a = area_a["safety_score"]
    score_b = area_b["safety_score"]
    tot_a = area_a["total_annual_crimes"]
    tot_b = area_b["total_annual_crimes"]

    if score_a > score_b:
        winner = "A"
        winner_name = area_a["name"]
        diff_score = score_a - score_b
        fewer_crimes_pct = round((tot_b - tot_a) / max(1, tot_b) * 100) if tot_b > tot_a else 0
        title = f"🏆 {area_a['name']} is Safer than {area_b['name']}"
        summary = (
            f"{area_a['name']} scored higher in overall safety ({score_a}/100 vs {score_b}/100)"
            + (f" and has {fewer_crimes_pct}% fewer reported crimes." if fewer_crimes_pct > 0 else ".")
        )
    elif score_b > score_a:
        winner = "B"
        winner_name = area_b["name"]
        diff_score = score_b - score_a
        fewer_crimes_pct = round((tot_a - tot_b) / max(1, tot_a) * 100) if tot_a > tot_b else 0
        title = f"🏆 {area_b['name']} is Safer than {area_a['name']}"
        summary = (
            f"{area_b['name']} scored higher in overall safety ({score_b}/100 vs {score_a}/100)"
            + (f" and has {fewer_crimes_pct}% fewer reported crimes." if fewer_crimes_pct > 0 else ".")
        )
    else:
        winner = "EQUAL"
        winner_name = "Both Areas"
        diff_score = 0
        title = f"🤝 Both areas have similar safety ratings ({score_a}/100)"
        summary = f"Both {area_a['name']} and {area_b['name']} show similar safety indicators."

    categories_comparison = [
        {"category": "Theft & Pickpocketing", "count_a": area_a["crime_breakdown"]["theft"], "count_b": area_b["crime_breakdown"]["theft"]},
        {"category": "Robbery", "count_a": area_a["crime_breakdown"]["robbery"], "count_b": area_b["crime_breakdown"]["robbery"]},
        {"category": "Violent Crimes", "count_a": area_a["crime_breakdown"]["murder"], "count_b": area_b["crime_breakdown"]["murder"]},
        {"category": "Women Safety Cases", "count_a": area_a["crime_breakdown"]["rape"], "count_b": area_b["crime_breakdown"]["rape"]},
        {"category": "Riots & Street Disputes", "count_a": area_a["crime_breakdown"]["riots"], "count_b": area_b["crime_breakdown"]["riots"]},
    ]

    return {
        "profile": {
            "key": profile_key,
            "label": profile["label"],
            "description": profile["description"],
        },
        "verdict": {
            "winner": winner,
            "winner_name": winner_name,
            "title": title,
            "summary": summary,
            "score_diff": diff_score,
        },
        "area_a": area_a,
        "area_b": area_b,
        "categories_comparison": categories_comparison,
        "profiles_available": {
            k: {"label": v["label"], "description": v["description"]}
            for k, v in TRAVEL_PROFILES.items()
        },
    }


# =============================================================================
# 7. Tactical Manpower & Predictive Resource Allocation Engine
# =============================================================================

def _generate_directive(primary_driver: str, tier: str) -> str:
    """Generate simple, actionable guidance based on primary threat driver and tier."""
    category_labels = {
        "murder": "violent crime and homicide hotspots",
        "rape": "women safety corridors & transit zones",
        "robbery": "transit routes & commercial market hubs",
        "riots": "sensitive public squares & crowded gathering points",
        "theft": "busy markets & vehicle parking clusters",
    }
    target = category_labels.get(primary_driver, "crime hotspots")

    if tier == "CRITICAL":
        return f"🚨 Urgent: Deploy 24/7 patrol squads targeting {target}."
    elif tier == "HIGH":
        return f"⚠️ High priority: Increase vehicle checkpoints and night patrols for {target}."
    elif tier == "ELEVATED":
        return f"⚡ Moderate: Strengthen regular foot beats and CCTV monitoring in {target}."
    return "✅ Normal: Maintain standard police station duty and routine neighborhood visits."


def allocate_manpower(
    state: str = "All India",
    target_year: int = 2027,
    total_manpower: int = 500,
    strategy_key: str = "balanced",
    min_per_entity: int = 5,
) -> Dict[str, Any]:
    """
    Predict future crime risk trajectories across districts (or states) and
    dynamically optimize tactical police manpower allocation to maximize harm reduction.

    Args:
        state: Target State/UT or 'All India'.
        target_year: Planning horizon year (2025–2035).
        total_manpower: Total personnel pool to distribute (50–50,000).
        strategy_key: Focus strategy ('balanced', 'violent', 'property', 'riots').
        min_per_entity: Minimum baseline reserve per district/entity (1–50).

    Returns:
        Structured allocation plan with risk tiers, threat indices, tactical directives,
        and comparative risk mitigation efficiency metrics.
    """
    try:
        target_year = max(2025, min(2035, int(target_year)))
    except (ValueError, TypeError):
        target_year = 2027

    try:
        total_manpower = max(50, min(50000, int(total_manpower)))
    except (ValueError, TypeError):
        total_manpower = 500

    try:
        min_per_entity = max(1, min(50, int(min_per_entity)))
    except (ValueError, TypeError):
        min_per_entity = 5

    strategy = STRATEGIES.get(strategy_key, STRATEGIES["balanced"])
    weights = strategy["weights"]

    is_all_india = not state or state.strip() == "All India"
    df = DATAFRAME

    if is_all_india:
        entity_col = "state"
        entity_label = "States & UTs"
    else:
        df = df[df.state == state.strip().upper()]
        entity_col = "district"
        entity_label = "Districts"

    if df.empty:
        return {
            "error": f"No historical records found for state '{state}'."
        }

    categories_list = list(weights.keys())
    grouped = (
        df.groupby([entity_col, "year"], as_index=False)[categories_list]
        .sum()
    )

    entities = sorted(grouped[entity_col].unique())
    if not entities:
        return {"error": "No geographic entities available for resource allocation."}

    rows: List[Dict[str, Any]] = []

    for ent in entities:
        e_df = grouped[grouped[entity_col] == ent].sort_values("year")
        if len(e_df) < 2:
            continue

        X = e_df[["year"]].values
        preds: Dict[str, float] = {}
        threat = 0.0

        for cat, weight in weights.items():
            y = e_df[cat].values
            if len(e_df) >= 3:
                model = LinearRegression().fit(X, y)
                pred_val = max(0.0, float(model.predict([[target_year]])[0]))
            else:
                pred_val = max(0.0, float(y.mean()))

            preds[cat] = round(pred_val, 1)
            threat += pred_val * weight

        primary_driver = max(
            categories_list,
            key=lambda c: preds.get(c, 0.0) * weights.get(c, 1.0)
        )

        rows.append({
            "entity": ent,
            "threat": round(threat, 1),
            "predicted_total": int(round(sum(preds.values()))),
            "breakdown": preds,
            "primary_driver": primary_driver,
        })

    if not rows:
        return {"error": "Insufficient historical data to project future risk for allocation."}

    rows.sort(key=lambda x: x["threat"], reverse=True)
    num_entities = len(rows)

    # 1. Calculate baseline reservation and discretionary pool
    base_allocation = min(min_per_entity, total_manpower // num_entities)
    base_pool_total = num_entities * base_allocation
    discretionary_pool = total_manpower - base_pool_total

    threat_values = np.array([r["threat"] for r in rows], dtype=float)
    total_threat = float(threat_values.sum())

    if total_threat > 0 and discretionary_pool > 0:
        added_officers = np.round(
            discretionary_pool * (threat_values / total_threat)
        ).astype(int)
    else:
        added_officers = np.zeros(num_entities, dtype=int)

    # Correct integer rounding variance so total matches total_manpower exactly
    variance = total_manpower - (base_pool_total + int(added_officers.sum()))
    added_officers[0] += variance

    # 2. Risk Tier Classification based on threat distribution
    p75 = float(np.percentile(threat_values, 75)) if num_entities > 3 else 0.0
    p50 = float(np.percentile(threat_values, 50)) if num_entities > 3 else 0.0
    p25 = float(np.percentile(threat_values, 25)) if num_entities > 3 else 0.0

    allocations: List[Dict[str, Any]] = []
    strategic_alloc_list: List[int] = []

    for rank, (r, add_count) in enumerate(zip(rows, added_officers), start=1):
        allocated = base_allocation + int(add_count)
        strategic_alloc_list.append(allocated)
        share_pct = round((allocated / total_manpower) * 100, 1)

        th = r["threat"]
        if num_entities <= 2:
            tier = "CRITICAL" if rank == 1 else "ELEVATED"
        elif th >= p75 and th > 0:
            tier = "CRITICAL"
        elif th >= p50 and th > 0:
            tier = "HIGH"
        elif th >= p25 and th > 0:
            tier = "ELEVATED"
        else:
            tier = "BASELINE"

        directive = _generate_directive(r["primary_driver"], tier)

        allocations.append({
            "rank": rank,
            "entity": r["entity"],
            "risk_tier": tier,
            "threat_index": r["threat"],
            "predicted_crimes": r["predicted_total"],
            "allocated_manpower": allocated,
            "share_pct": share_pct,
            "primary_driver": r["primary_driver"].capitalize(),
            "tactical_directive": directive,
            "breakdown": r["breakdown"],
        })

    # 3. Compute Risk Mitigation Efficiency Gain vs Uniform Allocation
    uniform_officers = total_manpower / num_entities
    exp_uniform = float(np.sum(threat_values / (1.0 + 0.04 * uniform_officers)))
    exp_strategic = float(np.sum(
        threat_values / (1.0 + 0.04 * np.array(strategic_alloc_list, dtype=float))
    ))
    mitigation_gain = (
        round(max(0.0, (1.0 - (exp_strategic / exp_uniform)) * 100), 1)
        if exp_uniform > 0
        else 0.0
    )

    critical_count = sum(1 for a in allocations if a["risk_tier"] == "CRITICAL")
    high_count = sum(1 for a in allocations if a["risk_tier"] == "HIGH")

    return {
        "state": "All India" if is_all_india else state.strip().upper(),
        "is_all_india": is_all_india,
        "entity_label": entity_label,
        "target_year": target_year,
        "total_manpower": total_manpower,
        "base_per_entity": base_allocation,
        "strategy": {
            "key": strategy_key,
            "label": strategy["label"],
            "description": strategy["description"],
        },
        "total_entities": num_entities,
        "total_threat_score": round(total_threat, 1),
        "total_predicted_crimes": int(round(float(sum(r["predicted_total"] for r in rows)))),
        "mitigation_improvement_pct": mitigation_gain,
        "critical_count": critical_count,
        "high_count": high_count,
        "top_hotspots": allocations[:5],
        "allocations": allocations,
        "strategies_available": {
            k: {"label": v["label"], "description": v["description"]}
            for k, v in STRATEGIES.items()
        },
    }

