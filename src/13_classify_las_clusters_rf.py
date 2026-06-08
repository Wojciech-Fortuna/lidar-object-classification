from pathlib import Path
import argparse

import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_DIR / "models" / "random_forest.pkl"


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


FEATURE_COLUMNS = [
    "num_points",
    "length",
    "width",
    "height",
    "bounding_box_volume",
    "point_density",
    "z_min",
    "z_max",
    "z_mean",
    "z_std",
    "intensity_mean",
    "intensity_std",
    "linearity",
    "planarity",
    "scattering",
    "omnivariance",
    "anisotropy",
    "eigenentropy",
    "verticality",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify LAS clusters using the DALES-trained Random Forest."
    )

    parser.add_argument(
        "las_stem",
        help="LAS file stem, e.g. cloud2"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    las_stem = Path(args.las_stem).stem

    features_file = (
        PROJECT_DIR /
        "data" /
        f"las_cluster_features_{las_stem}.csv"
    )

    output_file = (
        PROJECT_DIR /
        "data" /
        f"las_cluster_predictions_{las_stem}.csv"
    )

    if not features_file.exists():
        print(f"Features file not found: {features_file}")
        return

    print("Loading LAS cluster features...")
    df = pd.read_csv(features_file)

    print("Loading Random Forest model...")
    model = joblib.load(MODEL_FILE)

    X = df[FEATURE_COLUMNS]

    print("Running prediction...")
    predicted_ids = model.predict(X)

    result = df.copy()
    result["predicted_class_id"] = predicted_ids
    result["predicted_class"] = result["predicted_class_id"].map(CLASS_NAMES)

    print("\nPredictions:")
    print(
        result[
            [
                "cluster_id",
                "num_points",
                "linearity",
                "planarity",
                "scattering",
                "predicted_class_id",
                "predicted_class",
            ]
        ].to_string(index=False)
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved to:\n{output_file}")


if __name__ == "__main__":
    main()