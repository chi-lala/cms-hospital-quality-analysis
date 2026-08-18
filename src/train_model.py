from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("data/processed/model_data.csv")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the modeling dataset."""

    return pd.read_csv(DATA_PATH)


def split_data(df):
    """Split data into training and testing sets."""

    X = df.drop(columns=["high_rating"])

    y = df["high_rating"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def build_preprocessor():
    """Build preprocessing pipeline."""

    numeric_features = [
        "mort_better_rate",
        "mort_worse_rate",
        "safety_better_rate",
        "safety_worse_rate",
        "readm_better_rate",
        "readm_worse_rate",
    ]

    categorical_features = [
        "hospital_type",
        "hospital_ownership",
        "emergency_services",
        "state",
    ]

    numeric_pipeline = Pipeline([
        ("imputer",
            SimpleImputer(strategy="median")),

        ("scaler",
            StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer",
            SimpleImputer(strategy="most_frequent")),

        ("encoder",
            OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric",
            numeric_pipeline,
            numeric_features),

        ("categorical",
            categorical_pipeline,
            categorical_features)
    ])

    return preprocessor


def build_model():
    """Build logistic regression pipeline."""

    preprocessor = build_preprocessor()

    model = Pipeline([
        ("preprocessor",preprocessor),

        ("classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42)
        )
    ])

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

# Accuracy
    print(
        f"\nAccuracy: "
        f"{accuracy_score(y_test, predictions):.3f}"
    )

# Precision
    print(
        f"Precision: "
        f"{precision_score(y_test, predictions):.3f}"
    )

# Recall
    print(
        f"Recall: "
        f"{recall_score(y_test, predictions):.3f}"
    )

# F1 Score
    print(
        f"F1 Score: "
        f"{f1_score(y_test, predictions):.3f}"
    )

# ROC-AUC
    print(
        f"ROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.3f}"
    )

# Confusion Matrix
    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")

    print(classification_report(
            y_test,
            predictions)
    )


# Coefficient Analysis
def analyze_coefficients(model):
    """Analyze and visualize logistic regression coefficients."""

    print("\n" + "=" * 60)
    print("MODEL B: LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 60)

    # Get the preprocessing pipeline
    preprocessor = model.named_steps["preprocessor"]

    # Get the trained logistic regression classifier
    classifier = model.named_steps["classifier"]

    # Get feature names after preprocessing
    feature_names = (preprocessor.get_feature_names_out())

    coefficients = classifier.coef_[0]

    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients)
    })

    # Sort by absolute coefficient
    importance = importance.sort_values("abs_coefficient", ascending=False)

    output_csv = (PROCESSED_DIR / "model_feature_importance.csv")
    importance.to_csv(output_csv, index=False)

    print(f"Saved feature importance to: {output_csv}")

    print("\nTop 15 model features:")

    print(
        importance[["feature", "coefficient"]]
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Visuals
    # --------------------------------------------------

    top_features = (importance
        .head(15)
        .sort_values("coefficient")
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature"],
        top_features["coefficient"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    plt.title("Top Model Features — Logistic Regression")
    plt.tight_layout()

    output_path = (Path("outputs/figures") / "logistic_regression_feature_importance.png")

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved figure to: {output_path}")


# Logistic Regression Model
def build_characteristics_model():
    """Build a model using hospital characteristics only."""

    categorical_features = [
        "hospital_type",
        "hospital_ownership",
        "emergency_services",
        "state",
    ]

    categorical_pipeline = Pipeline([
        ("imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        ("encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        ("categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    model = Pipeline([
        ("preprocessor",
            preprocessor
        ),

        ("classifier",
            LogisticRegression(max_iter=1000, random_state=42)
        )
    ])

    return model


# Random Forest Model
def build_random_forest():
    """Build a Random Forest classification pipeline."""

    preprocessor = build_preprocessor()

    model = Pipeline([
        ("preprocessor",
            preprocessor
        ),

        ("classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ])

    return model


# Model comparison table
def calculate_metrics(model, X_test, y_test):
    """Calculate classification metrics for a model."""

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probability)
    }


# Model Prediction Export
def save_model_predictions(characteristics_model, full_model, random_forest, X_test, y_test):
    """Save test-set predictions from all models for Power BI."""

    actual = y_test.reset_index(drop=True)

    prediction_data = []

    models = {
        "Characteristics Only": characteristics_model,
        "Logistic Regression": full_model,
        "Random Forest": random_forest
    }

    for model_name, model in models.items():

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        model_results = pd.DataFrame({
            "actual": actual,
            "model": model_name,
            "prediction": predictions,
            "probability": probabilities
        })

        prediction_data.append(model_results)

    predictions_df = pd.concat(prediction_data,ignore_index=True)

    output_path = (PROCESSED_DIR / "model_predictions.csv")

    predictions_df.to_csv(output_path, index=False)

    print(f"Saved model predictions to: {output_path}")


# Model Coefficient Export
def get_logistic_regression_coefficients(model, model_name):
    """Extract logistic regression coefficients for export."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance = pd.DataFrame({
        "model": model_name,
        "feature": feature_names,
        "importance": coefficients,
        "abs_importance": abs(coefficients)    
    })

    importance["direction"] = np.where(importance["importance"] > 0, "positive", "negative")

    return importance

def get_random_forest_feature_importance(model):
    """Extract Random Forest feature importance for export."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_

    importance = pd.DataFrame({
        "model": "Random Forest",
        "feature": feature_names,
        "importance": importance,
        "abs_importance": abs(importance),
        "direction": "N/A"
    })

    return importance


# Model Comparison Plot
def plot_model_comparison(results):
    """Create a comparison chart for model performance."""

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):

        values = [
            results[model_name][metric]
            for metric in metrics
        ]

        ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=model_name
        )

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()

    output_path = ("outputs/figures/model_comparison.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {output_path}")


# Save results
def save_model_results(results):
    """Save model performance metrics for dashboard use."""

    results_df = pd.DataFrame(results).T.reset_index()

    results_df = results_df.rename(columns={"index": "model"})

    output_path = Path("data/processed/model_results.csv")

    results_df.to_csv(output_path, index=False)

    print(f"Saved model results to: {output_path}")


def main():

    print("Loading model dataset...")

    df = load_data()

    print(f"Loaded {len(df):,} records.")

    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Training rows: {len(X_train):,}")

    print(f"Testing rows: {len(X_test):,}")

    results = {}

    # --------------------------------------------------
    # MODEL A
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL A: HOSPITAL CHARACTERISTICS ONLY")
    print("=" * 60)

    characteristics_model = (build_characteristics_model())

    print("\nTraining Model A...")

    characteristics_model.fit(X_train, y_train)

    print("Model A training complete.")

    evaluate_model(characteristics_model,
        X_test,
        y_test
    )

    results["Characteristics Only"] = calculate_metrics(characteristics_model, X_test, y_test)

    # --------------------------------------------------
    # MODEL B
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL B: CHARACTERISTICS + QUALITY MEASURES")
    print("=" * 60)

    full_model = build_model()

    print("\nTraining Model B...")

    full_model.fit(X_train, y_train)

    print("Model B training complete.")

    evaluate_model(full_model, X_test, y_test)

    analyze_coefficients(full_model)

    results["Logistic Regression"] = calculate_metrics(full_model, X_test, y_test)


    # --------------------------------------------------
    # MODEL C
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL C: RANDOM FOREST")
    print("=" * 60)

    random_forest = build_random_forest()

    print("\nTraining Model C...")

    random_forest.fit(X_train,y_train)

    print("Model C training complete.")

    evaluate_model(random_forest, X_test, y_test)

    results["Random Forest"] = calculate_metrics(random_forest, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    plot_model_comparison(results)

    # Save results
    save_model_results(results)

    save_model_predictions(characteristics_model, full_model, random_forest, X_test, y_test)

    model_features = pd.concat([
        get_logistic_regression_coefficients(characteristics_model, "Characteristics Only"),
        get_logistic_regression_coefficients(full_model, "Logistic Regression"),
        get_random_forest_feature_importance(random_forest)
    ])

    output_path = (PROCESSED_DIR / "model_feature_importance.csv")
    model_features.to_csv(output_path, index=False)
    print(f"Saved model feature importance data to: {output_path}")

if __name__ == "__main__":
    main()