# CrimeLens — India Crime Analytics & Forecasting

A structured Flask intelligence dashboard focused on annual National Crime Record Bureau (NCRB) crime analysis, category comparisons, year-over-year shifts, and machine learning trend projections.

---

## 1. Features
- **Comprehensive NCRB Data**: Covers official district-level IPC crime statistics for 2001–2014, combined with demonstration projections through 2026.
- **Hierarchical Geographic Filtering**: Analyze crime at the All-India, State/UT, or individual District level.
- **Time Series & YoY Analytics**: Year-wise trend tracking for total crimes, murder, rape, theft, robbery, and riots with absolute and percentage change indicators.
- **Crime Category Breakdown**: Compare relative distributions across crime types.
- **Top Rankings Leaderboard**: Identify top state and district counts across years.
- **Machine Learning Forecasting**: Linear regression trend forecasting for 3 or 5 years with chronological holdout validation (MAE and RMSE metrics).
- **Responsive Dashboard**: Optimized for desktop, tablet, and mobile viewing.

---

## 2. Project Structure

```text
.
├── app.py                     # Flask web server and REST API endpoints
├── analytics.py               # Data processing, aggregation, and ML forecasting engine
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
├── data/                      # Active analytical datasets (2001–2026)
│   ├── 01_District_wise_crimes_committed_IPC_2001_2012.csv
│   ├── 01_District_wise_crimes_committed_IPC_2013.csv
│   ├── 01_District_wise_crimes_committed_IPC_2014.csv
│   └── SYNTHETIC_demo_crime_data_2015_2026.csv
├── Dataset/                   # Complete collection of 77 official NCRB datasets
├── templates/
│   └── index.html             # Semantic single-page dashboard template
└── static/
    ├── app.js                 # Modular frontend charts and data controller
    └── style.css              # Dashboard design tokens and responsive stylesheet
```

---

## 3. Installation & Local Setup

### Prerequisites
- Python 3.10+
- `pip` package manager

### Steps
1. Open terminal in the project root directory.

2. Create and activate a virtual environment:
   ```bash
   # Windows PowerShell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:
   ```bash
   python app.py
   ```

5. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 4. REST API Reference

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/` | `GET` | None | Serves the single-page dashboard HTML. |
| `/api/options` | `GET` | `state` (optional) | Returns available states, districts, years, and categories. |
| `/api/summary` | `GET` | `state`, `district` | Returns key metrics (total crimes, records, year span). |
| `/api/trends` | `GET` | `category`, `state`, `district` | Returns time-series counts with annual YoY change values. |
| `/api/breakdown` | `GET` | `state`, `district`, `year` | Returns aggregated counts by crime category. |
| `/api/forecast` | `GET` | `category`, `state`, `district`, `horizon` | Returns ML linear trend predictions with MAE/RMSE. |
| `/api/places` | `GET` | `state`, `category`, `year` | Returns top 10 locations ranked by recorded volume. |

---

## 5. Important Data & Modeling Notes

- **Data Normalization**: Aggregate rows (such as `TOTAL` and `ZZ TOTAL`) are filtered out during ingestion to prevent double-counting district records with state totals.
- **Reporting Schema Variations**: The 2014 NCRB schema transitioned to revised column naming conventions; the ingestion pipeline normalizes these variations into a unified schema.
- **Demonstration Extension**: Official NCRB source files cover 2001–2014; synthetic projection records extend the visual series through 2026 for demonstration purposes.
- **Predictive Modeling**: Trend forecasts employ a Linear Regression extrapolation with chronological holdout validation on the most recent observations. Forecasts are intended for planning and analytical demonstration, not operational policing guarantees.
