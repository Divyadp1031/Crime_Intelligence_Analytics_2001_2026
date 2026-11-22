
from app import DATAFRAME
import pandas as pd

df = DATAFRAME.copy()

df["rape"] = pd.to_numeric(
    df["rape"], errors="coerce"
).fillna(0)

def check_data(data, period):
    print("\n==========", period, "==========")

    total = data["rape"].sum()

    # Total grouped by state
    state_total = (
        data.groupby("state")["rape"].sum().sum()
    )

    # Total grouped by district
    district_total = (
        data.groupby(["state", "district"])["rape"]
        .sum()
        .sum()
    )

    print("Total recorded cases:", int(total))
    print("State-wise total:", int(state_total))
    print("District-wise total:", int(district_total))

    print("State difference:", int(total - state_total))
    print("District difference:", int(total - district_total))

    print(
        "Rows with missing state:",
        data["state"].isna().sum()
    )

    print(
        "Rows with missing district:",
        data["district"].isna().sum()
    )

    duplicates = data.duplicated(
        subset=["state", "district", "year"]
    ).sum()

    print("Duplicate district-year rows:", duplicates)


# Check all available years
check_data(df, "ALL YEARS")

# Check one individual year
check_data(
    df[df["year"] == 2014],
    "YEAR 2014"
)
