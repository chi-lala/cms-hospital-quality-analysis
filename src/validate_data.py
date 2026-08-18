from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/Hospital_General_Information.csv")


def load_data():
    """Load the raw CMS hospital dataset."""
    
    return pd.read_csv(DATA_PATH)


def inspect_dataset(df):
    """Print basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

# Data dimensions
    print("\nDataset Size:")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

# Column names
    print("\nColumn Names:")
    for column in df.columns:
        print(f"  - {column}")

# Data types
    print("\nData Types:")
    print(df.dtypes)

# Missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))

# Duplicates
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


def inspect_key_variables(df):
    """Inspect important variables and their unique values."""

    print("\n" + "=" * 60)
    print("KEY VARIABLE VALIDATION")
    print("=" * 60)

    # Facility ID uniqueness
    print("\nFacility ID:")
    print(f"  Unique IDs: {df['Facility ID'].nunique():,}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Duplicate IDs: {df['Facility ID'].duplicated().sum():,}")

    # Hospital type
    print("\nHospital Type:")
    print(df["Hospital Type"].value_counts(dropna=False))

    # Hospital ownership
    print("\nHospital Ownership:")
    print(df["Hospital Ownership"].value_counts(dropna=False))

    # Emergency services
    print("\nEmergency Services:")
    print(df["Emergency Services"].value_counts(dropna=False))

    # Overall hospital rating
    print("\nHospital Overall Rating:")
    print(df["Hospital overall rating"].value_counts(dropna=False))

    # Birthing friendly designation
    print("\nBirthing Friendly Designation:")
    print(
        df["Meets criteria for birthing friendly designation"]
        .value_counts(dropna=False)
    )

    # State
    print("\nState:")
    print(f"  Unique states/territories: {df['State'].nunique()}")
    print(df["State"].value_counts().head(10))


def inspect_measure_columns(df):
    """Inspect values in hospital quality measure count columns."""

    measure_columns = [
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
    ]

    print("\n" + "=" * 60)
    print("QUALITY MEASURE VALUE INSPECTION")
    print("=" * 60)

    for column in measure_columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(15))



def main():
    print("Loading CMS hospital dataset...")

    df = load_data()

    print("Dataset loaded successfully.")

    inspect_dataset(df)
    inspect_key_variables(df)
    inspect_measure_columns(df)


if __name__ == "__main__":
    main()