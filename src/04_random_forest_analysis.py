from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix


TRAIN_PATH = Path("data/features_train.csv")
TEST_PATH = Path("data/features_test.csv")

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)

CLASS_LABELS = [1, 2, 3, 4, 5, 6, 7, 8]

CLASS_NAMES = [
    "ground",
    "vegetation",
    "cars",
    "trucks",
    "power_lines",
    "fences",
    "poles",
    "buildings",
]


def main():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    columns_to_drop = [
        "file_name",
        "instance_id",
        "semantic_class",
    ]

    X_train = train_data.drop(columns=columns_to_drop)
    y_train = train_data["semantic_class"]

    X_test = test_data.drop(columns=columns_to_drop)
    y_test = test_data["semantic_class"]

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # =========================
    # Feature Importance
    # =========================

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    )

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False,
    )

    print("\nFeature Importance:\n")
    print(feature_importance)

    feature_importance.to_csv(
        RESULTS_PATH / "feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
    )

    plt.gca().invert_yaxis()

    plt.title("Random Forest Feature Importance")
    plt.tight_layout()

    plt.savefig(
        RESULTS_PATH / "feature_importance.png",
        dpi=300,
    )

    plt.show()

    # =========================
    # Confusion Matrix
    # =========================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_LABELS,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    disp.plot(
        xticks_rotation=45,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_PATH / "confusion_matrix.png",
        dpi=300,
    )

    plt.show()


if __name__ == "__main__":
    main()