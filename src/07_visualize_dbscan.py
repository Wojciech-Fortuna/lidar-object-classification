from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
from plyfile import PlyData


PROJECT_DIR = Path(__file__).resolve().parents[1]
PLY_FILE = PROJECT_DIR / "DALESObjects" / "train" / "5080_54435_new.ply"

GROUND_CLASS = 1

EPS = 0.5
MIN_POINTS = 20

RANSAC_DISTANCE_THRESHOLD = 0.45


def load_point_cloud(file_path):
    ply_data = PlyData.read(file_path)
    points = ply_data["testing"].data

    return pd.DataFrame(
        {
            "x": points["x"],
            "y": points["y"],
            "z": points["z"],
            "semantic_class": points["sem_class"],
            "instance_class": points["ins_class"],
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


def create_open3d_point_cloud(dataframe):
    xyz = dataframe[["x", "y", "z"]].values

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    return point_cloud


def run_dbscan(point_cloud):
    labels = np.array(
        point_cloud.cluster_dbscan(
            eps=EPS,
            min_points=MIN_POINTS,
            print_progress=True,
        )
    )

    return labels


def remove_ground_ransac(dataframe):
    xyz = dataframe[["x", "y", "z"]].values

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=0.45,
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

    return non_ground


def colorize_clusters(point_cloud, labels):
    max_label = labels.max()

    colors = np.zeros((len(labels), 3))

    if max_label >= 0:
        normalized_labels = labels / max_label
        colors = plt_colormap(normalized_labels)

    # Noise points are black
    colors[labels < 0] = [0, 0, 0]

    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def plt_colormap(values):
    """
    Simple colormap replacement without requiring matplotlib.
    Returns RGB colors for values in range [0, 1].
    """

    values = np.asarray(values)

    colors = np.zeros((len(values), 3))

    colors[:, 0] = np.sin(2 * np.pi * values) * 0.5 + 0.5
    colors[:, 1] = np.sin(2 * np.pi * values + 2.0) * 0.5 + 0.5
    colors[:, 2] = np.sin(2 * np.pi * values + 4.0) * 0.5 + 0.5

    return colors


def print_summary(labels):
    number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    number_of_noise_points = int(np.sum(labels == -1))

    print("\nDBSCAN summary")
    print("-" * 30)
    print(f"Clusters found: {number_of_clusters}")
    print(f"Noise points: {number_of_noise_points}")


def main():
    print("Loading point cloud...")

    dataframe = load_point_cloud(PLY_FILE)

    print(f"Original points: {len(dataframe):,}")

    dataframe = select_spatial_tile(dataframe)

    print(f"Points in selected spatial tile: {len(dataframe):,}")

    dataframe = remove_ground_ransac(dataframe)

    print(f"After RANSAC ground removal: {len(dataframe):,}")

    point_cloud = create_open3d_point_cloud(dataframe)

    print("Running DBSCAN...")

    labels = run_dbscan(point_cloud)

    print_summary(labels)

    print("Visualizing DBSCAN clusters...")

    colored_point_cloud = colorize_clusters(point_cloud, labels)

    o3d.visualization.draw_geometries(
        [colored_point_cloud],
        window_name="DBSCAN Clusters",
        width=1280,
        height=720,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()