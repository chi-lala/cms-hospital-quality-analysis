from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw/Hospital_General_Information.csv")

PROCESSED_DATA_PATH = Path("data/processed/hospital_quality_clean.csv")


NUMERIC_COLUMNS = [
    "ZIP Code",
    "MORT Group Measure Count",
    "Count of Facility MORT Measures",
    "Count of MORT Measures Better",
    "Count of MORT Measures No Different",
    "Count of MORT Measures Worse",
    "Safety Group Measure Count",
    "Count of Facility Safety Measures",
    "Count of Safety Measures Better",
    "Count of Safety Measures No Different",
    "Count of Safety Measures Worse",
    "READM Group Measure Count",
    "Count of Facility READM Measures",
    "Count of READM Measures Better",
    "Count of READM Measures No Different",
    "Count of READM Measures Worse",
    "Pt Exp Group Measure Count",
    "Count of Facility Pt Exp Measures",
    "TE Group Measure Count",
    "Count of Facility TE Measures",
    "Hospital overall rating",
]


def load_raw_data():
    """Load the raw CMS hospital dataset."""

    return pd.read_csv(RAW_DATA_PATH)


def convert_numeric_columns(df):
    """Convert numeric-looking columns to numeric values."""

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def standardize_column_names(df):
    """Create standardized snake_case column names for analysis."""

    df = df.rename(
        columns={
            "Facility ID": "facility_id",
            "Facility Name": "facility_name",
            "Address": "address",
            "City/Town": "city",
            "State": "state",
            "ZIP Code": "zip_code",
            "County/Parish": "county",
            "Telephone Number": "telephone",
            "Hospital Type": "hospital_type",
            "Hospital Ownership": "hospital_ownership",
            "Emergency Services": "emergency_services",
            "Meets criteria for birthing friendly designation":
                "birthing_friendly",
            "Hospital overall rating": "overall_rating",
            "Hospital overall rating footnote":
                "overall_rating_footnote",
        }
    )

    return df


def validate_measure_counts(df):
    """Check consistency of quality measure classification counts."""

    measure_groups = {
        "MORT": {
            "facility": "Count of Facility MORT Measures",
            "better": "Count of MORT Measures Better",
            "same": "Count of MORT Measures No Different",
            "worse": "Count of MORT Measures Worse",
        },
        "Safety": {
            "facility": "Count of Facility Safety Measures",
            "better": "Count of Safety Measures Better",
            "same": "Count of Safety Measures No Different",
            "worse": "Count of Safety Measures Worse",
        },
        "READM": {
            "facility": "Count of Facility READM Measures",
            "better": "Count of READM Measures Better",
            "same": "Count of READM Measures No Different",
            "worse": "Count of READM Measures Worse",
        },
    }

    print("\n" + "=" * 60)
    print("MEASURE COUNT INTEGRITY CHECK")
    print("=" * 60)

    for group, columns in measure_groups.items():

        expected = df[columns["facility"]]

        calculated = (
            df[columns["better"]]
            + df[columns["same"]]
            + df[columns["worse"]]
        )

        comparable = expected.notna() & calculated.notna()

        mismatches = (expected[comparable] != calculated[comparable]).sum()

        checked = comparable.sum()

        print(f"\n{group}:")
        print(f"Records checked: {checked:,}")
        print(f"Mismatches: {mismatches:,}")

        if mismatches == 0:
            print("Integrity check passed")
        else:
            print("Integrity check requires investigation")



def validate_data(df):
    """Run basic integrity checks on the cleaned dataset."""

    print("\n" + "=" * 60)
    print("CLEANED DATA VALIDATION")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    duplicate_ids = df["facility_id"].duplicated().sum()

    print(f"Duplicate facility IDs: {duplicate_ids:,}")

    if duplicate_ids != 0:
        raise ValueError(
            "Duplicate Facility IDs detected."
        )

    print("\nOverall Rating Distribution:")
    print(df["overall_rating"].value_counts(dropna=False).sort_index())

    print("\nMissing Values:")
    missing = df.isna().sum()
    print(
        missing[missing > 0]
        .sort_values(ascending=False)
        .head(15)
    )


def save_processed_data(df):
    """Save the cleaned dataset."""

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nProcessed dataset saved to: " f"{PROCESSED_DATA_PATH}")


def main():
    print("Loading raw CMS hospital data...")

    df = load_raw_data()

    print("Converting numeric columns...")
    df = convert_numeric_columns(df)

    print("Standardizing column names...")
    df = standardize_column_names(df)

    validate_data(df)

    validate_measure_counts(df)

    save_processed_data(df)

    print("\nCleaning pipeline complete.")


if __name__ == "__main__":
    main()