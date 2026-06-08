from pathlib import Path
import argparse

import numpy as np
import open3d as o3d
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


CLASS_COLORS = {
    "ground": [0.5, 0.5, 0.5],
    "vegetation": [0.0, 0.8, 0.0],
    "cars": [1.0, 0.0, 0.0],
    "trucks": [1.0, 0.5, 0.0],
    "power_lines": [0.0, 1.0, 1.0],
    "fences": [1.0, 0.0, 1.0],
    "poles": [1.0, 1.0, 0.0],
    "buildings": [0.0, 0.0, 1.0],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize LAS cluster predictions."
    )

    parser.add_argument(
        "las_stem",
        help="LAS file stem, e.g. cloud2"
    )

    return parser.parse_args()


def load_points_and_labels(points_labels_file):
    data = np.load(points_labels_file)

    points = data["points"]
    labels = data["labels"]

    points_centered = points - points.mean(axis=0)

    return points_centered, labels


def visualize(points, labels, predictions_file):
    predictions = pd.read_csv(predictions_file)

    cluster_to_class = dict(
        zip(
            predictions["cluster_id"],
            predictions["predicted_class"]
        )
    )

    colors = np.zeros((len(points), 3))

    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue

        mask = labels == cluster_id

        predicted_class = cluster_to_class.get(
            int(cluster_id),
            "vegetation"
        )

        color = CLASS_COLORS.get(
            predicted_class,
            [1.0, 1.0, 1.0]
        )

        colors[mask] = color

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    print("\nLegend:")
    for class_name, color in CLASS_COLORS.items():
        print(f"{class_name:12s} -> {color}")

    print("\nOpening Open3D viewer...")

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="DALES model predictions on LAS clusters",
        width=1400,
        height=800,
    )


def main():
    args = parse_args()

    las_stem = Path(args.las_stem).stem

    points_labels_file = (
        PROJECT_DIR /
        "data" /
        f"las_cluster_points_labels_{las_stem}.npz"
    )

    predictions_file = (
        PROJECT_DIR /
        "data" /
        f"las_cluster_predictions_{las_stem}.csv"
    )

    if not points_labels_file.exists():
        print(f"Points/labels file not found: {points_labels_file}")
        print("Run feature extraction first, for example:")
        print(f"python src/15_extract_las_cluster_features.py {las_stem}.las")
        return

    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        print("Run classification first, for example:")
        print(f"python src/17_classify_las_clusters_rf.py {las_stem}")
        return

    print("Loading points and DBSCAN labels...")
    points, labels = load_points_and_labels(points_labels_file)

    print("Visualizing predictions...")
    visualize(
        points,
        labels,
        predictions_file
    )


if __name__ == "__main__":
    main()