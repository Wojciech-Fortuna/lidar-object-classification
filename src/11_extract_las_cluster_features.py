from pathlib import Path
import argparse

import laspy
import numpy as np
import pandas as pd
import open3d as o3d


PROJECT_DIR = Path(__file__).resolve().parents[1]

ROI_SIZE = 100.0

GROUND_PERCENTILE = 5
MIN_HEIGHT_ABOVE_GROUND = 3.0

EPS = 0.8
MIN_POINTS = 30

MAX_POINTS_FOR_DBSCAN = 2_000_000
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract cluster features from a LAS file."
    )

    parser.add_argument(
        "las_file",
        help="LAS filename from the obloty directory, e.g. cloud2.las",
    )

    return parser.parse_args()


def load_roi(file_path, roi_size):
    print(f"Loading: {file_path}")

    las = laspy.read(file_path)

    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)
    intensity = np.asarray(las.intensity)

    x_center = np.median(x)
    y_center = np.median(y)

    half = roi_size / 2

    mask = (
        (x >= x_center - half)
        & (x <= x_center + half)
        & (y >= y_center - half)
        & (y <= y_center + half)
    )

    print(f"Points in ROI: {mask.sum():,}")

    return np.column_stack([
        x[mask],
        y[mask],
        z[mask],
        intensity[mask],
    ])


def remove_ground(points):
    ground_level = np.percentile(
        points[:, 2],
        GROUND_PERCENTILE
    )

    print(f"Estimated ground level: {ground_level:.2f} m")

    mask = points[:, 2] > ground_level + MIN_HEIGHT_ABOVE_GROUND

    filtered = points[mask]

    print(
        f"Points above ground: "
        f"{len(filtered):,} / {len(points):,}"
    )

    return filtered


def limit_points_for_dbscan(points):
    if len(points) <= MAX_POINTS_FOR_DBSCAN:
        return points

    print(
        f"Too many points for DBSCAN: {len(points):,}. "
        f"Sampling {MAX_POINTS_FOR_DBSCAN:,} points."
    )

    rng = np.random.default_rng(RANDOM_SEED)

    indices = rng.choice(
        len(points),
        MAX_POINTS_FOR_DBSCAN,
        replace=False
    )

    return points[indices]


def run_dbscan(points):
    xyz = points[:, :3]
    centered = xyz - xyz.mean(axis=0)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(centered)

    labels = np.array(
        cloud.cluster_dbscan(
            eps=EPS,
            min_points=MIN_POINTS,
            print_progress=True
        )
    )

    return labels


def empty_pca_features():
    return {
        "linearity": 0,
        "planarity": 0,
        "scattering": 0,
        "omnivariance": 0,
        "anisotropy": 0,
        "eigenentropy": 0,
        "verticality": 0,
    }


def compute_pca_features(xyz_points):
    if len(xyz_points) < 3:
        return empty_pca_features()

    centered = xyz_points - np.mean(xyz_points, axis=0)
    cov = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    l1, l2, l3 = eigenvalues

    if l1 <= 0:
        return empty_pca_features()

    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    scattering = l3 / l1
    anisotropy = (l1 - l3) / l1

    omnivariance = (
        (l1 * l2 * l3) ** (1 / 3)
        if l1 > 0 and l2 > 0 and l3 > 0
        else 0
    )

    normalized = eigenvalues / np.sum(eigenvalues)

    eigenentropy = -np.sum(
        normalized * np.log(normalized + 1e-12)
    )

    normal_vector = eigenvectors[:, 2]
    verticality = 1 - abs(normal_vector[2])

    return {
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "omnivariance": omnivariance,
        "anisotropy": anisotropy,
        "eigenentropy": eigenentropy,
        "verticality": verticality,
    }


def extract_cluster_features(points, labels):
    features = []

    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids >= 0]

    for cluster_id in cluster_ids:
        cluster_points = points[labels == cluster_id]

        xyz = cluster_points[:, :3]
        intensity = cluster_points[:, 3]

        num_points = len(cluster_points)

        min_xyz = xyz.min(axis=0)
        max_xyz = xyz.max(axis=0)

        length = max_xyz[0] - min_xyz[0]
        width = max_xyz[1] - min_xyz[1]
        height = max_xyz[2] - min_xyz[2]

        bounding_box_volume = length * width * height

        point_density = (
            num_points / bounding_box_volume
            if bounding_box_volume > 0
            else 0
        )

        z_values = xyz[:, 2]
        pca = compute_pca_features(xyz)

        features.append({
            "cluster_id": int(cluster_id),
            "num_points": int(num_points),
            "length": round(length, 3),
            "width": round(width, 3),
            "height": round(height, 3),
            "bounding_box_volume": round(bounding_box_volume, 3),
            "point_density": round(point_density, 6),
            "z_min": round(float(z_values.min()), 3),
            "z_max": round(float(z_values.max()), 3),
            "z_mean": round(float(z_values.mean()), 3),
            "z_std": round(float(z_values.std()), 3),
            "intensity_mean": round(float(intensity.mean()), 3),
            "intensity_std": round(float(intensity.std()), 3),
            "linearity": round(pca["linearity"], 4),
            "planarity": round(pca["planarity"], 4),
            "scattering": round(pca["scattering"], 4),
            "omnivariance": round(pca["omnivariance"], 4),
            "anisotropy": round(pca["anisotropy"], 4),
            "eigenentropy": round(pca["eigenentropy"], 4),
            "verticality": round(pca["verticality"], 4),
        })

    return pd.DataFrame(features)


def save_points_and_labels(points, labels, output_npz):
    xyz = points[:, :3].astype(np.float32)
    labels = labels.astype(np.int32)

    np.savez_compressed(
        output_npz,
        points=xyz,
        labels=labels,
    )

    print(f"\nSaved DBSCAN points and labels to:\n{output_npz}")


def main():
    args = parse_args()

    las_name = args.las_file
    las_stem = Path(las_name).stem

    las_file = PROJECT_DIR / "obloty" / las_name

    output_csv = (
        PROJECT_DIR /
        "data" /
        f"las_cluster_features_{las_stem}.csv"
    )

    output_npz = (
        PROJECT_DIR /
        "data" /
        f"las_cluster_points_labels_{las_stem}.npz"
    )

    if not las_file.exists():
        print(f"LAS file not found: {las_file}")
        return

    print("Loading ROI...")
    points = load_roi(las_file, ROI_SIZE)

    if len(points) == 0:
        print("No points found in ROI.")
        return

    print("\nRemoving ground...")
    points = remove_ground(points)

    if len(points) == 0:
        print("No points remaining after ground filtering.")
        return

    points = limit_points_for_dbscan(points)

    print("\nRunning DBSCAN...")
    print(f"EPS = {EPS}")
    print(f"MIN_POINTS = {MIN_POINTS}")
    labels = run_dbscan(points)

    print("\nExtracting cluster features...")
    df = extract_cluster_features(points, labels)

    df = df.sort_values(
        "linearity",
        ascending=False
    )

    print("\nClusters sorted by linearity:")
    print(df.head(20).to_string(index=False))

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False
    )

    save_points_and_labels(
        points,
        labels,
        output_npz
    )

    print(f"\nSaved features to:\n{output_csv}")


if __name__ == "__main__":
    main()