import re
import pandas as pd
import numpy as np
from scipy.spatial import KDTree, ConvexHull
import networkx as nx
from hilbertcurve.hilbertcurve import HilbertCurve
from tqdm import tqdm
from collections import Counter
from networkx.algorithms.community import louvain_communities


TILE_CSV = "/path/to/tile_cluster_labels.csv"
OUTPUT_FILE = "/path/to/output_directory/wsi_level_features.csv"

IMG_SIZE = 224
R_ADJ = IMG_SIZE * 1.5
HILBERT_P = 10
HILBERT_N = 2
N_CLUSTERS = 23


df = pd.read_csv(TILE_CSV)


def parse_xy(path):
    match = re.search(
        r"tile_x(\d+)_y(\d+)_",
        path
    )

    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y

    return 0, 0


df[["x", "y"]] = df["tile_path"].apply(
    lambda path: pd.Series(parse_xy(path))
)


def safe_convex_hull(points):
    if (
        len(points) < 3
        or np.ptp(points[:, 0]) == 0
        or np.ptp(points[:, 1]) == 0
    ):
        return 0.0, 0.0

    try:
        hull = ConvexHull(
            points,
            qhull_options="QJ"
        )

        polygon = hull.points[
            hull.vertices
        ]

        area = 0.5 * np.abs(
            np.dot(
                polygon[:, 0],
                np.roll(
                    polygon[:, 1],
                    -1
                )
            )
            - np.dot(
                polygon[:, 1],
                np.roll(
                    polygon[:, 0],
                    -1
                )
            )
        )

        perimeter = np.sum(
            np.linalg.norm(
                np.diff(
                    polygon,
                    axis=0
                ),
                axis=1
            )
        )

        perimeter += np.linalg.norm(
            polygon[0]
            - polygon[-1]
        )

        return area, perimeter

    except Exception:
        return 0.0, 0.0


def compute_wsi_features(
    wsi_df,
    n_clusters=N_CLUSTERS
):
    features = {}

    coords = wsi_df[
        ["x", "y"]
    ].values

    clusters = wsi_df[
        "leiden_1"
    ].values

    counts = Counter(
        clusters
    )

    total = len(
        clusters
    )

    for cluster_id in range(
        n_clusters
    ):
        features[
            f"cluster_{cluster_id}_prop"
        ] = (
            counts.get(
                cluster_id,
                0
            )
            / total
        )

    tree = KDTree(
        coords
    )

    scm = np.zeros(
        (
            n_clusters,
            n_clusters
        )
    )

    for idx, cluster_id in enumerate(
        clusters
    ):
        neighbors = tree.query_ball_point(
            coords[idx],
            R_ADJ
        )

        for neighbor in neighbors:
            scm[
                cluster_id,
                clusters[neighbor]
            ] += 1

    scm = scm / (
        scm.sum(
            axis=1,
            keepdims=True
        )
        + 1e-8
    )

    for i in range(
        n_clusters
    ):
        for j in range(
            i,
            n_clusters
        ):
            features[
                f"SCM_{i}_{j}"
            ] = scm[
                i,
                j
            ]

    graph = nx.Graph()

    for cluster_id in range(
        n_clusters
    ):
        graph.add_node(
            cluster_id
        )

    for i in range(
        n_clusters
    ):
        for j in range(
            i + 1,
            n_clusters
        ):
            if scm[i, j] > 0:
                graph.add_edge(
                    i,
                    j,
                    weight=scm[i, j]
                )

    degrees = dict(
        graph.degree(
            weight="weight"
        )
    )

    for cluster_id in range(
        n_clusters
    ):
        features[
            f"graph_deg_{cluster_id}"
        ] = degrees.get(
            cluster_id,
            0
        )

    clustering = nx.clustering(
        graph,
        weight="weight"
    )

    for cluster_id in range(
        n_clusters
    ):
        features[
            f"graph_cluster_{cluster_id}"
        ] = clustering.get(
            cluster_id,
            0
        )

    try:
        degree_values = np.array(
            list(
                degrees.values()
            )
        )

        if degree_values.sum() > 0:
            probability = (
                degree_values
                / degree_values.sum()
            )
        else:
            probability = np.zeros_like(
                degree_values
            )

        features[
            "graph_entropy"
        ] = -np.sum(
            [
                value * np.log(value)
                for value in probability
                if value > 0
            ]
        )

    except Exception:
        features[
            "graph_entropy"
        ] = 0

    try:
        communities = louvain_communities(
            graph,
            weight="weight"
        )

        features[
            "graph_modularity"
        ] = nx.community.modularity(
            graph,
            communities,
            weight="weight"
        )

    except Exception:
        features[
            "graph_modularity"
        ] = 0

    for cluster_id in range(
        n_clusters
    ):
        points = coords[
            clusters == cluster_id
        ]

        area, perimeter = safe_convex_hull(
            points
        )

        features[
            f"cluster_{cluster_id}_hull_area"
        ] = area

        features[
            f"cluster_{cluster_id}_hull_perimeter"
        ] = perimeter

    max_coord = coords.max(
        axis=0
    )

    max_coord[
        max_coord == 0
    ] = 1

    coords_norm = (
        coords
        / max_coord
        * (
            2 ** HILBERT_P
            - 1
        )
    ).astype(
        int
    )

    hilbert = HilbertCurve(
        HILBERT_P,
        HILBERT_N
    )

    hilbert_index = np.array(
        [
            hilbert.distance_from_point(
                coord.tolist()
            )
            for coord in coords_norm
        ]
    )

    sorted_idx = np.argsort(
        hilbert_index
    )

    sorted_clusters = clusters[
        sorted_idx
    ]

    tpm = np.zeros(
        (
            n_clusters,
            n_clusters
        )
    )

    for idx in range(
        len(sorted_clusters) - 1
    ):
        tpm[
            sorted_clusters[idx],
            sorted_clusters[idx + 1]
        ] += 1

    tpm = tpm / (
        tpm.sum(
            axis=1,
            keepdims=True
        )
        + 1e-8
    )

    for i in range(
        n_clusters
    ):
        for j in range(
            n_clusters
        ):
            features[
                f"TPM_{i}_{j}"
            ] = tpm[
                i,
                j
            ]

    return features


wsi_features = []

for wsi_id, group in tqdm(
    df.groupby("wsi_id")
):
    features = compute_wsi_features(
        group,
        n_clusters=N_CLUSTERS
    )

    features[
        "wsi_id"
    ] = wsi_id

    wsi_features.append(
        features
    )


wsi_features_df = pd.DataFrame(
    wsi_features
)

wsi_features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "WSI-level feature matrix saved."
)