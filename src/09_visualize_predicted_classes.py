from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import open3d as o3d
from plyfile import PlyData


PROJECT_DIR = Path(__file__).resolve().parents[1]

PLY_FILE = PROJECT_DIR / "DALESObjects" / "train" / "5080_54435_new.ply"
MODEL_PATH = PROJECT_DIR / "models" / "random_forest.pkl"

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


CLASS_COLORS = {
    2: [0.0, 0.8, 0.0],
    3: [1.0, 0.0, 0.0],
    4: [1.0, 0.5, 0.0],
    5: [1.0, 1.0, 0.0],
    6: [0.0, 1.0, 1.0],
    7: [1.0, 0.0, 1.0],
    8: [0.2, 0.2, 1.0],
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


def select_spatial_tile(df):
    x_min = df["x"].quantile(0.45)
    x_max = df["x"].quantile(0.55)

    y_min = df["y"].quantile(0.45)
    y_max = df["y"].quantile(0.55)

    return df[
        (df["x"] >= x_min)
        & (df["x"] <= x_max)
        & (df["y"] >= y_min)
        & (df["y"] <= y_max)
    ].copy()


def remove_ground_ransac(df):
    xyz = df[["x", "y", "z"]].values

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=1000,
    )

    ground_mask = np.zeros(len(df), dtype=bool)
    ground_mask[inliers] = True

    non_ground = df.loc[~ground_mask].copy()
    ground = df.loc[ground_mask].copy()

    print(f"RANSAC ground points: {len(ground):,}")
    print(f"RANSAC non-ground points: {len(non_ground):,}")
    print(f"Plane model: {plane_model}")

    return non_ground


def run_dbscan(df):
    xyz = df[["x", "y", "z"]].values

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    labels = np.array(
        pcd.cluster_dbscan(
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


def extract_features(cluster_points):
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
    df = load_point_cloud(PLY_FILE)

    df = select_spatial_tile(df)

    print(
        "Ground truth ground points:",
        (df["semantic_class"] == GROUND_CLASS).sum()
    )

    df = remove_ground_ransac(df)
    df = df.reset_index(drop=True)

    print(f"Points used: {len(df):,}")

    labels = run_dbscan(df)

    df["cluster_id"] = labels

    colors = np.zeros((len(df), 3))

    for cluster_id in sorted(df["cluster_id"].unique()):

        if cluster_id == -1:
            continue

        cluster_points = df[df["cluster_id"] == cluster_id]

        if len(cluster_points) < MIN_POINTS:
            continue

        features = extract_features(cluster_points)

        feature_vector = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS,
        )

        prediction = model.predict(feature_vector)[0]

        cluster_color = CLASS_COLORS.get(
            prediction,
            [0.5, 0.5, 0.5],
        )

        colors[cluster_points.index] = cluster_color

    colors[df["cluster_id"] == -1] = [0, 0, 0]

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        df[["x", "y", "z"]].values
    )

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("\nClass legend:")
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 1:
            continue
        print(f"{class_id}: {class_name}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Predicted Classes - RANSAC + PCA",
        width=1400,
        height=900,
    )


if __name__ == "__main__":
    main()