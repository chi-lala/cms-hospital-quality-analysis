import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency


DATA_PATH = Path("data/processed/model_data.csv")


def load_data():
    """Load the modeling dataset."""

    print("Loading model dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df):,} records.")

    return df


def chi_square_test(df,feature):
    """Run a chi-square test for a categorical feature."""

    contingency_table = pd.crosstab(df[feature], df["high_rating"])

    chi2, p_value, dof, expected = (chi2_contingency(contingency_table))

    effect_size = cramers_v(contingency_table,chi2)

    print(f"Cramer's V: {effect_size:.3f}")

    print("\n" + "=" * 60)
    print(f"CHI-SQUARE TEST: {feature.upper()}")
    print("=" * 60)

    print("\nContingency table:")
    print(contingency_table)

    print(f"\nChi-square statistic: {chi2:.3f}")

    print(f"Degrees of freedom: {dof}")

    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("\nResult: Evidence of an association between the variables.")
    else:
        print("\nResult: Insufficient evidence of an association.")

    return {
        "feature": feature,
        "chi2": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "cramers_v": effect_size
    }


def cramers_v(contingency_table, chi2):
    """Calculate Cramer's V effect size."""

    n = contingency_table.to_numpy().sum()

    rows, cols = contingency_table.shape

    minimum_dimension = min(rows - 1, cols - 1)

    return (chi2 / (n * minimum_dimension)) ** 0.5



def main():

    df = load_data()

    features = [
        "hospital_type",
        "hospital_ownership",
        "emergency_services",
        "state"
    ]

    results = []

    for feature in features:

        result = chi_square_test(df,feature)
        results.append(result)

    results_df = pd.DataFrame(results)

    # Statistical significance
    results_df["significant"] = results_df["p_value"] < 0.05

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 60)

    print(results_df.to_string(index=False))

    # Export results for dashboard
    output_path = Path("data/processed/statistical_results.csv")

    results_df.to_csv(output_path, index = False)

    print(f"\nSaved statistical results to: " f"{output_path}")

if __name__ == "__main__":
    main()