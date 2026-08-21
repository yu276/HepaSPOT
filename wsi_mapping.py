import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


CSV_PATH = "/path/to/tile_cluster_labels.csv"
OUTPUT_DIR = "/path/to/wsi_mapping_output"

df = pd.read_csv(CSV_PATH)

print("Columns:", df.columns)

cluster_col = "leiden_1"

clusters = sorted(
    df[cluster_col]
    .astype(int)
    .unique()
)

print("Clusters:", clusters)


all_colors = (
    list(plt.cm.tab20.colors)
    + list(plt.cm.tab20b.colors)
    + list(plt.cm.tab20c.colors)
)

cluster_color = {
    cluster: tuple(
        int(value * 255)
        for value in all_colors[
            cluster % len(all_colors)
        ]
    )
    for cluster in clusters
}

print(
    "Cluster-color mapping generated:",
    len(cluster_color)
)


def draw_cluster_legend(
    cluster_color,
    save_path="cluster_color_legend.png"
):
    num_clusters = len(cluster_color)

    fig, ax = plt.subplots(
        figsize=(6, num_clusters * 0.3)
    )

    for idx, (cluster, rgb) in enumerate(
        cluster_color.items()
    ):
        ax.barh(
            y=idx,
            width=1,
            color=[
                value / 255
                for value in rgb
            ]
        )

        ax.text(
            1.05,
            idx,
            f"Cluster {cluster}",
            va="center",
            fontsize=10
        )

    ax.set_ylim(
        -1,
        num_clusters
    )

    ax.set_axis_off()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(
        "Cluster color legend saved:",
        save_path
    )


draw_cluster_legend(
    cluster_color,
    save_path=os.path.join(
        OUTPUT_DIR,
        "cluster_color_legend.png"
    )
)


def parse_tile_coordinates(tile_path):
    name = os.path.basename(
        tile_path
    )

    match = re.search(
        r"x(\d+)_y(\d+)_w(\d+)_h(\d+)",
        name
    )

    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        w = int(match.group(3))
        h = int(match.group(4))

        return x, y, w, h

    raise ValueError(
        f"Unable to parse tile coordinates: {name}"
    )


def visualize_wsi(
    wsi_id,
    df,
    cluster_color,
    save_dir=OUTPUT_DIR
):
    os.makedirs(
        save_dir,
        exist_ok=True
    )

    subset = df[
        df["wsi_id"] == wsi_id
    ].copy()

    xs = []
    ys = []
    ws = []
    hs = []

    for path in subset[
        "tile_path"
    ]:
        x, y, w, h = parse_tile_coordinates(
            path
        )

        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)

    subset["x"] = xs
    subset["y"] = ys
    subset["w"] = ws
    subset["h"] = hs

    max_w = (
        subset["x"].max()
        + subset["w"].max()
    )

    max_h = (
        subset["y"].max()
        + subset["h"].max()
    )

    image = Image.new(
        "RGB",
        (
            max_w,
            max_h
        ),
        color=(
            255,
            255,
            255
        )
    )

    draw = ImageDraw.Draw(
        image
    )

    for _, row in subset.iterrows():
        cluster = int(
            row[cluster_col]
        )

        color = cluster_color[
            cluster
        ]

        draw.rectangle(
            [
                row["x"],
                row["y"],
                row["x"] + row["w"],
                row["y"] + row["h"]
            ],
            fill=color
        )

    save_path = os.path.join(
        save_dir,
        f"{wsi_id}_cluster_map.png"
    )

    image.save(
        save_path
    )

    print(
        "Saved:",
        save_path
    )


target_wsis = [
    "example_wsi_01",
    "example_wsi_02",
    "example_wsi_03",
    "example_wsi_04"
]

print(
    "Generating cluster maps for:",
    target_wsis
)


for wsi_id in target_wsis:
    visualize_wsi(
        wsi_id,
        df,
        cluster_color
    )