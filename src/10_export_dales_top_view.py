from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData


PROJECT_DIR = Path(__file__).resolve().parents[1]

PLY_FILE = (
    PROJECT_DIR /
    "DALESObjects" /
    "train" /
    "5080_54435_new.ply"
)

OUTPUT_FILE = (
    PROJECT_DIR /
    "results" /
    "dales_5080_54435_top_view.png"
)


CLASS_COLORS = {
    1: [0.5, 0.5, 0.5],  # ground
    2: [0.0, 0.8, 0.0],  # vegetation
    3: [1.0, 0.0, 0.0],  # cars
    4: [1.0, 0.5, 0.0],  # trucks
    5: [0.0, 1.0, 1.0],  # power_lines
    6: [1.0, 0.0, 1.0],  # fences
    7: [1.0, 1.0, 0.0],  # poles
    8: [0.0, 0.0, 1.0],  # buildings
}


def main():
    print(f"Loading: {PLY_FILE}")

    ply = PlyData.read(PLY_FILE)
    vertex = ply["testing"].data

    x = np.asarray(vertex["x"])
    y = np.asarray(vertex["y"])
    labels = np.asarray(vertex["sem_class"])

    colors = np.zeros((len(labels), 3))

    for class_id, color in CLASS_COLORS.items():
        colors[labels == class_id] = color

    print(f"Points: {len(x):,}")
    print("Classes found:", sorted(np.unique(labels).tolist()))

    plt.figure(figsize=(12, 12))

    plt.scatter(
        x,
        y,
        c=colors,
        s=0.2,
        marker="."
    )

    plt.axis("equal")
    plt.axis("off")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print(f"\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()