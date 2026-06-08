from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
from plyfile import PlyData


PROJECT_DIR = Path(__file__).resolve().parents[1]
PLY_FILE = PROJECT_DIR / "DALESObjects" / "train" / "5080_54435_new.ply"

OUTPUT_PATH = PROJECT_DIR / "results"
OUTPUT_PATH.mkdir(exist_ok=True)

GROUND_CLASS = 1

EPS = 0.5
MIN_POINTS = 20

RANSAC_DISTANCE_THRESHOLD = 0.45


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

    ground = dataframe.loc[ground_mask].copy()
    non_ground = dataframe.loc[~ground_mask].copy()

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

    print("\nRANSAC ground removal")
    print("-" * 30)
    print(f"Ground truth ground points: {true_ground.sum():,}")
    print(f"RANSAC ground points: {len(ground):,}")
    print(f"RANSAC non-ground points: {len(non_ground):,}")
    print(f"Plane model: {plane_model}")
    print(f"Ground precision: {precision:.4f}")
    print(f"Ground recall:    {recall:.4f}")
    print(f"Ground F1-score:  {f1:.4f}")

    return non_ground, ground


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


def analyze_clusters(dataframe):
    rows = []

    clusters = sorted(dataframe["cluster_id"].unique())

    for cluster_id in clusters:
        if cluster_id == -1:
            continue

        cluster_points = dataframe[dataframe["cluster_id"] == cluster_id]
        cluster_size = len(cluster_points)

        semantic_counts = cluster_points["semantic_class"].value_counts()
        dominant_semantic_class = int(semantic_counts.idxmax())
        dominant_semantic_count = int(semantic_counts.max())

        instance_counts = cluster_points["instance_class"].value_counts()
        dominant_instance_class = int(instance_counts.idxmax())
        dominant_instance_count = int(instance_counts.max())

        semantic_purity = dominant_semantic_count / cluster_size
        instance_purity = dominant_instance_count / cluster_size

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "dominant_semantic_class": dominant_semantic_class,
                "dominant_semantic_name": CLASS_NAMES.get(
                    dominant_semantic_class,
                    "unknown",
                ),
                "semantic_purity": semantic_purity,
                "dominant_instance_class": dominant_instance_class,
                "instance_purity": instance_purity,
            }
        )

    return pd.DataFrame(rows)


def main():
    print("Loading point cloud...")

    dataframe = load_point_cloud(PLY_FILE)

    print(f"Original points: {len(dataframe):,}")

    dataframe = select_spatial_tile(dataframe)

    print(f"Points in selected spatial tile: {len(dataframe):,}")

    dataframe, ground = remove_ground_ransac(dataframe)

    print(f"After RANSAC ground removal: {len(dataframe):,}")

    print("Running DBSCAN...")

    labels = run_dbscan(dataframe)

    dataframe["cluster_id"] = labels

    number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    number_of_noise_points = int(np.sum(labels == -1))

    print("\nDBSCAN summary")
    print("-" * 30)
    print(f"Clusters found: {number_of_clusters}")
    print(f"Noise points: {number_of_noise_points}")

    cluster_analysis = analyze_clusters(dataframe)

    output_file = OUTPUT_PATH / "dbscan_cluster_analysis_ransac.csv"
    cluster_analysis.to_csv(output_file, index=False)

    print(f"\nSaved cluster analysis to: {output_file}")

    print("\nLargest clusters:")
    print(
        cluster_analysis
        .sort_values("cluster_size", ascending=False)
        .head(20)
    )

    print("\nAverage semantic purity:")
    print(cluster_analysis["semantic_purity"].mean())

    print("\nAverage instance purity:")
    print(cluster_analysis["instance_purity"].mean())


if __name__ == "__main__":
    main()