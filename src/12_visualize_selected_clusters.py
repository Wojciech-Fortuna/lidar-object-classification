from pathlib import Path
import argparse

import laspy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]

ROI_SIZE = 100.0

GROUND_PERCENTILE = 5
MIN_HEIGHT_ABOVE_GROUND = 3.0

EPS = 0.8
MIN_POINTS = 30


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "las_file",
        help="LAS filename, e.g. cloud2.las"
    )

    parser.add_argument(
        "clusters",
        nargs="+",
        type=int,
        help="Cluster IDs to visualize"
    )

    return parser.parse_args()


def load_roi(file_path):
    las = laspy.read(file_path)

    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    x_center = np.median(x)
    y_center = np.median(y)

    half = ROI_SIZE / 2

    mask = (
        (x >= x_center - half) &
        (x <= x_center + half) &
        (y >= y_center - half) &
        (y <= y_center + half)
    )

    return np.column_stack([
        x[mask],
        y[mask],
        z[mask]
    ])


def remove_ground(points):
    ground_level = np.percentile(
        points[:, 2],
        GROUND_PERCENTILE
    )

    mask = (
        points[:, 2]
        > ground_level + MIN_HEIGHT_ABOVE_GROUND
    )

    return points[mask]


def run_dbscan(points):
    centered = points - points.mean(axis=0)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(centered)

    labels = np.array(
        cloud.cluster_dbscan(
            eps=EPS,
            min_points=MIN_POINTS,
            print_progress=True
        )
    )

    return centered, labels


def visualize(points, labels):
    colors = np.zeros((len(points), 3))

    cmap = plt.get_cmap("tab10")

    for i, cluster_id in enumerate(TARGET_CLUSTERS):
        mask = labels == cluster_id

        if np.any(mask):
            colors[mask] = cmap(i % 10)[:3]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    print("\nShowing clusters:")
    print(TARGET_CLUSTERS)

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="Selected clusters",
        width=1280,
        height=720
    )


def main():
    args = parse_args()

    las_file = (
        PROJECT_DIR /
        "obloty" /
        args.las_file
    )

    global TARGET_CLUSTERS
    TARGET_CLUSTERS = args.clusters

    print(f"LAS file: {las_file}")
    print(f"Clusters: {TARGET_CLUSTERS}")

    points = load_roi(las_file)
    points = remove_ground(points)

    centered_points, labels = run_dbscan(points)

    visualize(centered_points, labels)


if __name__ == "__main__":
    main()