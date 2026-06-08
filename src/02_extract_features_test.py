from pathlib import Path

import numpy as np
import pandas as pd
from plyfile import PlyData
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "DALESObjects" / "train"
OUTPUT_PATH = PROJECT_DIR / "data"

OUTPUT_PATH.mkdir(exist_ok=True)

MIN_POINTS_PER_OBJECT = 50
EPSILON = 1e-12


def calculate_pca_features(object_points):
    xyz = object_points[["x", "y", "z"]].values.astype(float)

    if len(xyz) < 3:
        return {
            "linearity": 0,
            "planarity": 0,
            "scattering": 0,
            "omnivariance": 0,
            "anisotropy": 0,
            "eigenentropy": 0,
            "verticality": 0,
        }

    xyz_centered = xyz - xyz.mean(axis=0)

    covariance_matrix = np.cov(xyz_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    eigenvalues = np.maximum(eigenvalues, EPSILON)

    lambda_1, lambda_2, lambda_3 = eigenvalues

    eigenvalue_sum = lambda_1 + lambda_2 + lambda_3
    normalized_eigenvalues = eigenvalues / eigenvalue_sum

    linearity = (lambda_1 - lambda_2) / lambda_1
    planarity = (lambda_2 - lambda_3) / lambda_1
    scattering = lambda_3 / lambda_1
    omnivariance = (lambda_1 * lambda_2 * lambda_3) ** (1 / 3)
    anisotropy = (lambda_1 - lambda_3) / lambda_1

    eigenentropy = -np.sum(
        normalized_eigenvalues
        * np.log(normalized_eigenvalues + EPSILON)
    )

    main_direction = eigenvectors[:, 0]
    verticality = 1 - abs(main_direction[2])

    return {
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "omnivariance": omnivariance,
        "anisotropy": anisotropy,
        "eigenentropy": eigenentropy,
        "verticality": verticality,
    }


def extract_features_from_ply(file_path):
    """Extract geometric features from all object instances in a PLY file."""

    ply_data = PlyData.read(file_path)
    point_cloud = ply_data["testing"].data

    dataframe = pd.DataFrame(
        {
            "x": point_cloud["x"],
            "y": point_cloud["y"],
            "z": point_cloud["z"],
            "intensity": point_cloud["intensity"],
            "semantic_class": point_cloud["sem_class"],
            "instance_class": point_cloud["ins_class"],
        }
    )

    dataframe = dataframe[dataframe["instance_class"] > 0]

    feature_rows = []

    for instance_id, object_points in dataframe.groupby("instance_class"):

        if len(object_points) < MIN_POINTS_PER_OBJECT:
            continue

        semantic_class = (
            object_points["semantic_class"]
            .mode()
            .iloc[0]
        )

        x_min, x_max = object_points["x"].min(), object_points["x"].max()
        y_min, y_max = object_points["y"].min(), object_points["y"].max()
        z_min, z_max = object_points["z"].min(), object_points["z"].max()

        object_length = x_max - x_min
        object_width = y_max - y_min
        object_height = z_max - z_min

        bounding_box_volume = object_length * object_width * object_height
        number_of_points = len(object_points)

        point_density = (
            number_of_points / bounding_box_volume
            if bounding_box_volume > 0
            else 0
        )

        pca_features = calculate_pca_features(object_points)

        feature_rows.append(
            {
                "file_name": file_path.name,
                "instance_id": int(instance_id),
                "semantic_class": int(semantic_class),
                "num_points": number_of_points,
                "length": object_length,
                "width": object_width,
                "height": object_height,
                "bounding_box_volume": bounding_box_volume,
                "point_density": point_density,
                "z_min": z_min,
                "z_max": z_max,
                "z_mean": object_points["z"].mean(),
                "z_std": object_points["z"].std(),
                "intensity_mean": object_points["intensity"].mean(),
                "intensity_std": object_points["intensity"].std(),
                "linearity": pca_features["linearity"],
                "planarity": pca_features["planarity"],
                "scattering": pca_features["scattering"],
                "omnivariance": pca_features["omnivariance"],
                "anisotropy": pca_features["anisotropy"],
                "eigenentropy": pca_features["eigenentropy"],
                "verticality": pca_features["verticality"],
            }
        )

    return feature_rows


def main():
    all_features = []

    ply_files = sorted(DATA_PATH.glob("*.ply"))
    ply_files = [
        file_path
        for file_path in ply_files
        if not file_path.name.startswith("._")
    ]

    for file_path in tqdm(ply_files, desc="Extracting features"):
        all_features.extend(
            extract_features_from_ply(file_path)
        )

    features_dataframe = pd.DataFrame(all_features)

    output_file = OUTPUT_PATH / "features_test.csv"

    features_dataframe.to_csv(output_file, index=False)

    print(f"Saved feature dataset to: {output_file}")
    print(f"Number of extracted objects: {len(features_dataframe)}")

    print("\nFirst five records:")
    print(features_dataframe.head())

    print("\nColumns:")
    print(features_dataframe.columns.tolist())

    print("\nClass distribution:")
    print(
        features_dataframe["semantic_class"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()