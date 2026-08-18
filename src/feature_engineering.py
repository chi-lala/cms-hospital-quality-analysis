from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/hospital_quality_clean.csv")

OUTPUT_PATH = Path("data/processed/hospital_quality_features.csv")


def load_data():
    """Load cleaned hospital data."""

    return pd.read_csv(INPUT_PATH)


def create_quality_features(df):
    """Create derived quality performance metrics."""

    df = df.copy()

    # Mortality
    mort_total = (
        df["Count of MORT Measures Better"]
        + df["Count of MORT Measures No Different"]
        + df["Count of MORT Measures Worse"]
    )

    df["mort_better_rate"] = (
        df["Count of MORT Measures Better"]
        / mort_total
    )

    df["mort_worse_rate"] = (
        df["Count of MORT Measures Worse"]
        / mort_total
    )

    # Safety
    safety_total = (
        df["Count of Safety Measures Better"]
        + df["Count of Safety Measures No Different"]
        + df["Count of Safety Measures Worse"]
    )

    df["safety_better_rate"] = (
        df["Count of Safety Measures Better"]
        / safety_total
    )

    df["safety_worse_rate"] = (
        df["Count of Safety Measures Worse"]
        / safety_total
    )

    # Readmissions
    readm_total = (
        df["Count of READM Measures Better"]
        + df["Count of READM Measures No Different"]
        + df["Count of READM Measures Worse"]
    )

    df["readm_better_rate"] = (
        df["Count of READM Measures Better"]
        / readm_total
    )

    df["readm_worse_rate"] = (
        df["Count of READM Measures Worse"]
        / readm_total
    )

    return df


def validate_features(df):
    """Validate newly created features."""

    feature_columns = [
        "mort_better_rate",
        "mort_worse_rate",
        "safety_better_rate",
        "safety_worse_rate",
        "readm_better_rate",
        "readm_worse_rate",
    ]

    print("\n" + "=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)

    for column in feature_columns:

        print(f"\n{column}")

        print(
            f"  Non-null values: "
            f"{df[column].notna().sum():,}"
        )

        print(f"  Min: {df[column].min():.3f}")

        print(f"  Max: {df[column].max():.3f}")

        print(f"  Mean: {df[column].mean():.3f}")


def main():

    print("Loading cleaned hospital data...")

    df = load_data()

    print(f"Loaded {len(df):,} records.")

    print("Creating quality performance features...")

    df = create_quality_features(df)

    validate_features(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFeature dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()