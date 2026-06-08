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


def load_point_cloud(file_path):
    ply_data = PlyData.read(file_path)
    points = ply_data["testing"].data

    dataframe = pd.DataFrame(
        {
            "x": points["x"],
            "y": points["y"],
            "z": points["z"],
            "semantic_class": points["sem_class"],
            "instance_class": points["ins_class"],
        }
    )

    return dataframe


def select_spatial_tile(dataframe):
    x_min = dataframe["x"].quantile(0.45)
    x_max = dataframe["x"].quantile(0.55)

    y_min = dataframe["y"].quantile(0.45)
    y_max = dataframe["y"].quantile(0.55)

    tile = dataframe[
        (dataframe["x"] >= x_min)
        & (dataframe["x"] <= x_max)
        & (dataframe["y"] >= y_min)
        & (dataframe["y"] <= y_max)
    ]

    return tile


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

def print_cluster_statistics(labels):
    number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    number_of_noise_points = np.sum(labels == -1)

    print("\nDBSCAN results")
    print("-" * 30)
    print(f"Clusters found: {number_of_clusters}")
    print(f"Noise points: {number_of_noise_points}")

    cluster_sizes = []

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        cluster_size = np.sum(labels == cluster_id)
        cluster_sizes.append(cluster_size)

    cluster_sizes = sorted(cluster_sizes, reverse=True)

    print("\nLargest clusters:")

    for index, size in enumerate(cluster_sizes[:20]):
        print(f"{index + 1:2d}: {size}")


def main():
    print("Loading point cloud...")

    dataframe = load_point_cloud(PLY_FILE)

    print(f"Original points: {len(dataframe):,}")

    dataframe = select_spatial_tile(dataframe)

    print(f"Points in selected spatial tile: {len(dataframe):,}")

    dataframe = remove_ground_ransac(dataframe)

    print(f"After RANSAC ground removal: {len(dataframe):,}")

    print("Running DBSCAN...")

    labels = run_dbscan(dataframe)

    print_cluster_statistics(labels)


if __name__ == "__main__":
    main()