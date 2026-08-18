from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/processed/hospital_quality_features.csv")

FIGURES_PATH = Path("outputs/figures")


def load_data():
    """Load the cleaned hospital dataset."""

    return pd.read_csv(DATA_PATH)


def plot_rating_distribution(df):
    """Plot the distribution of CMS overall hospital ratings."""

    rated_hospitals = df.dropna(
        subset=["overall_rating"]
    ).copy()

    rating_counts = (
        rated_hospitals["overall_rating"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(rating_counts.index.astype(str),rating_counts.values)

    plt.title("CMS Hospital Overall Rating Distribution")
    plt.xlabel("Overall Hospital Rating")
    plt.ylabel("Number of Hospitals")

    plt.tight_layout()

    output_path = (FIGURES_PATH / "hospital_rating_distribution.png")

    plt.savefig(output_path, dpi=300)

    plt.show()

    print(f"Saved figure to: {output_path}")


def plot_hospitals_by_state(df):
    """Plot the number of hospitals by state."""

    state_counts = (
        df["state"]
        .value_counts()
        .head(15)
        .sort_values()
    )

    plt.figure(figsize=(9, 6))

    plt.barh(state_counts.index, state_counts.values)

    plt.title("Top 15 States by Number of Hospitals")
    plt.xlabel("Number of Hospitals")
    plt.ylabel("State")

    plt.tight_layout()

    output_path = (FIGURES_PATH / "hospitals_by_state.png")

    plt.savefig(output_path, dpi=300)

    plt.show()

    print(f"Saved figure to: {output_path}")


def plot_rating_distribution_by_hospital_type(df):
    """Plot the proportion of hospital ratings by hospital type."""

    rated_hospitals = df.dropna(
        subset=["overall_rating"]
    ).copy()

    # Only include hospital types with at least 30 rated hospitals
    type_counts = (
        rated_hospitals["hospital_type"]
        .value_counts()
    )

    valid_types = type_counts[
        type_counts >= 30
    ].index

    filtered = rated_hospitals[
        rated_hospitals["hospital_type"].isin(valid_types)
    ]

    # Calculate the proportion of each rating within each hospital type
    rating_distribution = pd.crosstab(
        filtered["hospital_type"],
        filtered["overall_rating"],
        normalize="index"
    )

    # Sort hospital types by their median rating
    median_ratings = (
        filtered
        .groupby("hospital_type")["overall_rating"]
        .median()
        .sort_values()
    )

    rating_distribution = rating_distribution.loc[median_ratings.index]

    # Create chart
    ax = rating_distribution.plot(
        kind="barh",
        stacked=True,
        figsize=(11, 7)
    )

    ax.set_title("CMS Hospital Rating Distribution by Hospital Type")
    ax.set_xlabel("Proportion of Hospitals")
    ax.set_ylabel("Hospital Type")

    ax.legend(
        title="Overall Rating",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()

    output_path = (FIGURES_PATH / "rating_distribution_by_hospital_type.png")

    plt.savefig(output_path, dpi=300)

    plt.show()

    print(f"Saved figure to: {output_path}")


def inspect_rating_outliers(df):
    """Identify potential rating outliers by hospital type."""

    rated_hospitals = df.dropna(
        subset=["overall_rating"]
    ).copy()

    for hospital_type, group in rated_hospitals.groupby("hospital_type"):
        if len(group) < 30:
            continue

        q1 = group["overall_rating"].quantile(0.25)
        q3 = group["overall_rating"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = group[
            (group["overall_rating"] < lower_bound)
            | (group["overall_rating"] > upper_bound)
        ]

        if len(outliers) > 0:
            print(f"\nHospital Type: {hospital_type}")
            print(f"Sample size: {len(group)}")
            print(f"Q1: {q1}")
            print(f"Q3: {q3}")
            print(f"IQR: {iqr}")
            print(f"Lower bound: {lower_bound}")
            print(f"Upper bound: {upper_bound}")
            print(f"Potential outliers: {len(outliers)}")

            print(outliers[
                    ["facility_id",
                        "facility_name",
                        "state",
                        "overall_rating"]
                ]
            )


def plot_rating_by_emergency_services(df):
    """Plot hospital rating distribution by emergency services."""

    rated_hospitals = df.dropna(
        subset=["overall_rating"]
    ).copy()

    rating_distribution = pd.crosstab(
        rated_hospitals["emergency_services"],
        rated_hospitals["overall_rating"],
        normalize="index"
    )

    ax = rating_distribution.plot(
        kind="bar",
        stacked=True,
        figsize=(9, 6)
    )

    ax.set_title("CMS Hospital Rating Distribution by Emergency Services")
    ax.set_xlabel("Emergency Services")
    ax.set_ylabel("Proportion of Hospitals")

    ax.legend(
        title="Overall Rating",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.xticks(rotation=0)

    plt.tight_layout()

    output_path = (FIGURES_PATH / "rating_by_emergency_services.png")

    plt.savefig(output_path, dpi=300)

    plt.show()

    print(f"Saved figure to: {output_path}")

def plot_quality_performance_rates(df):
    """Compare average better and worse rates across quality domains."""

    feature_columns = [
        "mort_better_rate",
        "mort_worse_rate",
        "safety_better_rate",
        "safety_worse_rate",
        "readm_better_rate",
        "readm_worse_rate",
    ]

    # Create a summary of mean rates
    summary = pd.DataFrame({
        "Quality Area": [
            "Mortality",
            "Safety",
            "Readmissions"
        ],
        "Better": [
            df["mort_better_rate"].mean(),
            df["safety_better_rate"].mean(),
            df["readm_better_rate"].mean()
        ],
        "Worse": [
            df["mort_worse_rate"].mean(),
            df["safety_worse_rate"].mean(),
            df["readm_worse_rate"].mean()
        ]
    })

    summary = summary.set_index("Quality Area")

    ax = summary.plot(kind="bar", figsize=(9, 6))

    ax.set_title("Average Hospital Quality Performance Rates")
    ax.set_xlabel("Quality Area")
    ax.set_ylabel("Average Proportion of Measures")

    ax.legend(title="Classification")

    plt.xticks(rotation=0)

    plt.tight_layout()

    output_path = (FIGURES_PATH / "quality_performance_rates.png")

    plt.savefig(output_path, dpi=300)

    plt.show()

    print(f"Saved figure to: {output_path}")


def main():
    print("Loading cleaned hospital data...")

    df = load_data()

    print(f"Loaded {len(df):,} hospital records.")

    FIGURES_PATH.mkdir(parents=True,exist_ok=True)

    print("\nCreating rating distribution...")
    plot_rating_distribution(df)

    print("\nCreating state distribution...")
    plot_hospitals_by_state(df)

    print("\nCreating rating distribution by hospital type...")
    plot_rating_distribution_by_hospital_type(df)

    print("\nCreating rating distribution by emergency services...")
    plot_rating_by_emergency_services(df)

    print("\nCreating quality performance rates comparison...")
    plot_quality_performance_rates(df)

    inspect_rating_outliers(df)

    print("\nEDA complete.")


if __name__ == "__main__":
    main()