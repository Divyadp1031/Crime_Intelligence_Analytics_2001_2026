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
