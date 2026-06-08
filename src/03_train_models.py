from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TRAIN_PATH = Path("data/features_train.csv")
TEST_PATH = Path("data/features_test.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    1: "ground",
    2: "vegetation",
    3: "cars",
    4: "trucks",
    5: "power_lines",
    6: "fences",
    7: "poles",
    8: "buildings",
}


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

    print("Features used for training:")
    for column in X_train.columns:
        print(f"- {column}")

    print(f"\nNumber of features: {X_train.shape[1]}")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
        ),
        "SVM": SVC(
            class_weight="balanced",
        ),
    }

    results = []

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(model_name)

        if model_name in ["KNN", "Logistic Regression", "SVM"]:
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
            }
        )

        print(f"Accuracy: {accuracy:.4f}")

        print(
            classification_report(
                y_test,
                predictions,
                labels=list(CLASS_NAMES.keys()),
                target_names=list(CLASS_NAMES.values()),
                zero_division=0,
            )
        )

        if model_name == "Random Forest":
            joblib.dump(
                model,
                MODEL_DIR / "random_forest.pkl"
            )

    joblib.dump(
        scaler,
        MODEL_DIR / "standard_scaler.pkl"
    )

    results_dataframe = pd.DataFrame(results)
    results_dataframe = results_dataframe.sort_values(
        by="accuracy",
        ascending=False,
    )

    print("\nFinal results:")
    print(results_dataframe)


if __name__ == "__main__":
    main()