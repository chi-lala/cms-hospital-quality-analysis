from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/hospital_quality_features.csv")

OUTPUT_PATH = Path("data/processed/model_data.csv")


def load_data():
    """Load the feature dataset."""

    return pd.read_csv(INPUT_PATH)


def create_target(df):
    """Create binary high-rating target."""

    df = df.copy()

    # Keep only hospitals with an available overall rating
    df = df.dropna(subset=["overall_rating"])

    # High rating = 4 or 5 stars
    df["high_rating"] = (df["overall_rating"] >= 4).astype(int)

    return df


def select_model_columns(df):
    """Select variables for modeling."""

    columns = [
        "facility_id",
        "facility_name",
        "high_rating",
        "hospital_type",
        "hospital_ownership",
        "emergency_services",
        "state",
        "mort_better_rate",
        "mort_worse_rate",
        "safety_better_rate",
        "safety_worse_rate",
        "readm_better_rate",
        "readm_worse_rate",
    ]

    return df[columns]


def validate_model_data(df):
    """Validate the modeling dataset."""

    print("\n" + "=" * 60)
    print("MODEL DATA VALIDATION")
    print("=" * 60)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nTarget distribution:")

    print(df["high_rating"]
        .value_counts()
        .sort_index()
    )

    print("\nTarget proportions:")

    print(df["high_rating"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nMissing values:")

    print(df.isna()
        .sum()
        .sort_values(ascending=False)
    )


def main():

    print("Loading feature dataset...")

    df = load_data()
    print(f"Loaded {len(df):,} records.")

    print("Creating target variable...")
    df = create_target(df)

    print("Selecting modeling variables...")
    df = select_model_columns(df)

    validate_model_data(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nModel dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()