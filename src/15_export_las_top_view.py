from pathlib import Path
import argparse

import laspy
import numpy as np
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]

MAX_POINTS = 2_000_000
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RGB top view from a LAS point cloud."
    )

    parser.add_argument(
        "las_file",
        help="LAS filename from the obloty directory, e.g. cloud2.las",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    las_name = args.las_file
    las_stem = Path(las_name).stem

    las_file = PROJECT_DIR / "obloty" / las_name
    output_file = (
        PROJECT_DIR /
        "results" /
        f"{las_stem}_top_view.png"
    )

    if not las_file.exists():
        print(f"LAS file not found: {las_file}")
        return

    print(f"Loading: {las_file}")

    las = laspy.read(las_file)

    x = np.asarray(las.x)
    y = np.asarray(las.y)

    red = np.asarray(las.red)
    green = np.asarray(las.green)
    blue = np.asarray(las.blue)

    n_points = len(x)

    print(f"Points: {n_points:,}")

    if n_points > MAX_POINTS:
        print(
            f"Random sampling: "
            f"{MAX_POINTS:,} points"
        )

        rng = np.random.default_rng(RANDOM_SEED)

        idx = rng.choice(
            n_points,
            MAX_POINTS,
            replace=False
        )

        x = x[idx]
        y = y[idx]

        red = red[idx]
        green = green[idx]
        blue = blue[idx]

    colors = np.column_stack(
        [
            red,
            green,
            blue,
        ]
    ).astype(np.float32)

    max_color = colors.max()

    if max_color > 0:
        colors /= max_color

    plt.figure(figsize=(14, 10))

    plt.scatter(
        x,
        y,
        c=colors,
        s=0.2,
        marker="."
    )

    plt.axis("equal")
    plt.axis("off")

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print(f"\nSaved:")
    print(output_file)


if __name__ == "__main__":
    main()