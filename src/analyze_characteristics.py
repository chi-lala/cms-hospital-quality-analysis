import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path("data/processed/model_data.csv")

OUTPUT_DIR = Path("outputs/figures")


def load_data():
    """Load the modeling dataset."""

    print("Loading model dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df):,} records.")

    return df


def analyze_category(df,column):
    """Calculate high-rating rates by category."""

    summary = (
        df.groupby(column)["high_rating"]
        .agg(
            hospitals="count",
            high_rated="sum",
            high_rating_rate="mean"
        )
        .reset_index()
    )

    summary["high_rating_rate"] *= 100

    summary = summary.sort_values("high_rating_rate",ascending=False)

    print("\n" + "=" * 60)
    print(f"{column.upper()} ANALYSIS")
    print("=" * 60)

    print(summary.to_string(index=False))

    return summary


def create_bar_chart(summary,category,filename):
    """Create a bar chart of high-rating rates."""

    plt.figure(figsize=(10, 6))
    plt.bar(summary[category],summary["high_rating_rate"])
    plt.ylabel("High-Rating Rate (%)")
    plt.xlabel(category.replace("_", " ").title())
    plt.title("High-Rating Rate by "
        + category.replace("_", " ").title())

    plt.xticks(rotation=45,ha="right")
    plt.tight_layout()

    output_path = (OUTPUT_DIR / filename)

    plt.savefig(output_path,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")


def analyze_state(df):
    """Calculate high-rating rates by state."""

    summary = (
    df.groupby("state")["high_rating"]
        .agg(
            hospitals="count",
            high_rated="sum",
            high_rating_rate="mean"
        )
        .reset_index()
    )

    summary["high_rating_rate"] *= 100

    # Only display states with at least 20 hospitals
    filtered = summary[summary["hospitals"] >= 20
    ].sort_values(
        "high_rating_rate",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("STATE ANALYSIS")
    print("=" * 60)

    print(filtered.to_string(index=False))

    return filtered



def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    categories = [
        "hospital_type",
        "hospital_ownership",
        "emergency_services"
    ]

    for category in categories:

        summary = analyze_category(df,category)

        filename = (f"{category}_high_rating.png")

        create_bar_chart(summary, category, filename)

    state_summary = analyze_state(df)

    create_bar_chart(state_summary, "state", "state_high_rating.png")


if __name__ == "__main__":
    main()