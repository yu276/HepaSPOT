import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns


FEATURES_FILE = "/path/to/tile_features.npy"
PATHS_FILE = "/path/to/tile_paths.npy"
OUTPUT_DIR = "/path/to/output_directory"

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_NEIGHBORS = 250
RESOLUTION = 0.5


def extract_wsi_id(tile_path):
    parent = os.path.dirname(tile_path)
    return os.path.basename(parent) if parent else os.path.basename(tile_path)


features = np.load(FEATURES_FILE)
paths = np.load(PATHS_FILE, allow_pickle=True)

assert features.shape[0] == len(paths), (
    "The number of feature vectors must match the number of tile paths."
)

print(
    f"Loaded {features.shape[0]} tiles, "
    f"feature dimension = {features.shape[1]}"
)


obs = pd.DataFrame({
    "tile_path": paths,
    "wsi_id": [
        extract_wsi_id(path)
        for path in paths
    ]
})

adata = anndata.AnnData(
    X=features,
    obs=obs.astype("category")
)


sc.tl.pca(
    adata,
    svd_solver="arpack",
    n_comps=min(
        50,
        features.shape[1] - 1
    )
)


sc.pp.neighbors(
    adata,
    n_neighbors=N_NEIGHBORS,
    n_pcs=min(
        adata.X.shape[1] - 1,
        50
    ),
    method="umap",
    metric="euclidean",
    key_added="nn_leiden"
)


groupby = f"leiden_{RESOLUTION}"

sc.tl.leiden(
    adata,
    resolution=RESOLUTION,
    key_added=groupby,
    neighbors_key="nn_leiden"
)


adata.obs[
    [
        groupby,
        "tile_path",
        "wsi_id"
    ]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "leiden_labels.csv"
    ),
    index=False
)


adata.write(
    os.path.join(
        OUTPUT_DIR,
        "tiles_leiden.h5ad"
    ),
    compression="gzip"
)


sc.tl.umap(
    adata,
    neighbors_key="nn_leiden"
)


sns.set_theme(
    style="white"
)

colors = sns.color_palette(
    "tab20",
    len(
        adata.obs[groupby].unique()
    )
)


fig = plt.figure(
    figsize=(10, 10)
)

ax = fig.add_subplot(
    1,
    1,
    1
)


sc.pl.umap(
    adata,
    color=groupby,
    legend_loc="on data",
    frameon=False,
    show=False,
    ax=ax,
    palette=colors,
    size=10
)


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "umap_leiden.png"
    ),
    dpi=300
)

plt.close()


print("UMAP plot saved.")
print("KNN and Leiden clustering completed.")