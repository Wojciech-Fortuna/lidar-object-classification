from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import open3d as o3d

from plyfile import PlyData


PLY_FILE = Path(
    r"C:\Users\Wojciech Fortuna\IWIUM\Project\DALESObjects\train\5080_54435_new.ply"
)

MODEL_PATH = Path(
    r"C:\Users\Wojciech Fortuna\IWIUM\Project\models\random_forest.pkl"
)

GROUND_CLASS = 1

EPS = 0.5
MIN_POINTS = 20

RANSAC_DISTANCE_THRESHOLD = 0.45
EPSILON = 1e-12


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


def load_point_cloud(file_path):
    ply_data = PlyData.read(file_path)
    points = ply_data["testing"].data

    return pd.DataFrame(
        {
            "x": points["x"],
            "y": points["y"],
            "z": points["z"],
            "intensity": points["intensity"],
            "semantic_class": points["sem_class"],
        }
    )


def select_spatial_tile(dataframe):
    x_min = dataframe["x"].quantile(0.45)
    x_max = dataframe["x"].quantile(0.55)

    y_min = dataframe["y"].quantile(0.45)
    y_max = dataframe["y"].quantile(0.55)

    return dataframe[
        (dataframe["x"] >= x_min)
        & (dataframe["x"] <= x_max)
        & (dataframe["y"] >= y_min)
        & (dataframe["y"] <= y_max)
    ].copy()


def remove_ground_ransac(dataframe):
    xyz = dataframe[["x", "y", "z"]].values

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=1000,
    )

    ground_mask = np.zeros(len(dataframe), dtype=bool)
    ground_mask[inliers] = True

    non_ground = dataframe.loc[~ground_mask].copy()
    ground = dataframe.loc[ground_mask].copy()

    print(f"RANSAC ground points: {len(ground):,}")
    print(f"RANSAC non-ground points: {len(non_ground):,}")
    print(f"Plane model: {plane_model}")

    true_ground = dataframe["semantic_class"] == GROUND_CLASS
    pred_ground = ground_mask

    tp = np.sum(true_ground & pred_ground)
    fp = np.sum(~true_ground & pred_ground)
    fn = np.sum(true_ground & ~pred_ground)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"Ground precision: {precision:.4f}")
    print(f"Ground recall:    {recall:.4f}")
    print(f"Ground F1-score:  {f1:.4f}")

    return non_ground


def run_dbscan(dataframe):
    xyz = dataframe[["x", "y", "z"]].values

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    labels = np.array(
        point_cloud.cluster_dbscan(
            eps=EPS,
            min_points=MIN_POINTS,
            print_progress=True,
        )
    )

    return labels


def calculate_pca_features(cluster_points):
    xyz = cluster_points[["x", "y", "z"]].values.astype(float)

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


def extract_cluster_features(cluster_points):
    x_min = cluster_points["x"].min()
    x_max = cluster_points["x"].max()

    y_min = cluster_points["y"].min()
    y_max = cluster_points["y"].max()

    z_min = cluster_points["z"].min()
    z_max = cluster_points["z"].max()

    length = x_max - x_min
    width = y_max - y_min
    height = z_max - z_min

    volume = length * width * height

    num_points = len(cluster_points)

    density = (
        num_points / volume
        if volume > 0
        else 0
    )

    pca_features = calculate_pca_features(cluster_points)

    return {
        "num_points": num_points,
        "length": length,
        "width": width,
        "height": height,
        "bounding_box_volume": volume,
        "point_density": density,
        "z_min": z_min,
        "z_max": z_max,
        "z_mean": cluster_points["z"].mean(),
        "z_std": cluster_points["z"].std(),
        "intensity_mean": cluster_points["intensity"].mean(),
        "intensity_std": cluster_points["intensity"].std(),
        "linearity": pca_features["linearity"],
        "planarity": pca_features["planarity"],
        "scattering": pca_features["scattering"],
        "omnivariance": pca_features["omnivariance"],
        "anisotropy": pca_features["anisotropy"],
        "eigenentropy": pca_features["eigenentropy"],
        "verticality": pca_features["verticality"],
    }


def main():
    print("Loading model...")

    model = joblib.load(MODEL_PATH)

    print("Loading point cloud...")

    dataframe = load_point_cloud(PLY_FILE)

    dataframe = select_spatial_tile(dataframe)

    print(
        "Ground truth ground points:",
        (dataframe["semantic_class"] == GROUND_CLASS).sum()
    )

    dataframe = remove_ground_ransac(dataframe)

    print(
        f"Points used: {len(dataframe):,}"
    )

    print("Running DBSCAN...")

    labels = run_dbscan(dataframe)

    dataframe["cluster_id"] = labels

    cluster_predictions = []

    for cluster_id in sorted(dataframe["cluster_id"].unique()):

        if cluster_id == -1:
            continue

        cluster_points = dataframe[
            dataframe["cluster_id"] == cluster_id
        ]

        if len(cluster_points) < MIN_POINTS:
            continue

        features = extract_cluster_features(cluster_points)

        feature_vector = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS,
        )

        prediction = model.predict(feature_vector)[0]

        cluster_predictions.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(cluster_points),
                "predicted_class": CLASS_NAMES[prediction],
            }
        )

    predictions_df = pd.DataFrame(cluster_predictions)

    predictions_df = predictions_df.sort_values(
        by="size",
        ascending=False,
    )

    print("\nLargest classified clusters")
    print("-" * 40)

    print(predictions_df.head(30))


if __name__ == "__main__":
    main()