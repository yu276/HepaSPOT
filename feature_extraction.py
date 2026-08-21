import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

from uni import get_encoder


CSV_FILE = "/path/to/tile_paths.csv"
CHECKPOINT = "/path/to/checkpoint/epoch_030.pth"
OUTPUT_DIR = "/path/to/output_directory"

IMG_SIZE = 224
FEATURE_DIM = 1536
PROJ_DIM = 128
BATCH_SIZE = 128
NUM_WORKERS = 8
ALPHA_PE = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Using device: {device}")


class TileDataset(Dataset):
    def __init__(self, df, uni_preproc):
        self.df = df
        self.preproc = uni_preproc
        self.resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["tile_path"]

        img = Image.open(path).convert("RGB")
        img = self.resize(img)
        img = self.preproc(img)

        basename = os.path.basename(path)
        match = re.search(
            r"tile_x(\d+)_y(\d+)_w\d+_h\d+",
            basename
        )

        if match:
            x = float(match.group(1))
            y = float(match.group(2))
        else:
            x = 0.0
            y = 0.0

        pe = torch.tensor(
            [x, y],
            dtype=torch.float32
        )

        return img, pe, path


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim=FEATURE_DIM,
        output_dim=PROJ_DIM
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class VICRegL_Uni(nn.Module):
    def __init__(self, uni_model, projector):
        super().__init__()

        self.uni = uni_model
        self.projector = projector
        self.pe_linear = nn.Linear(
            2,
            FEATURE_DIM
        )

    def forward(self, x, pe):
        with torch.no_grad():
            feat = self.uni(x)

        pe_feat = self.pe_linear(pe)

        feat = (
            feat
            + ALPHA_PE * pe_feat
        )

        proj = self.projector(feat)

        return proj


def load_model():
    print("[INFO] Loading UNI model...")

    uni_model, uni_preproc = get_encoder(
        enc_name="uni2-h"
    )

    uni_model.eval()

    for parameter in uni_model.parameters():
        parameter.requires_grad = False

    uni_model.to(device)

    projector = ProjectionHead(
        input_dim=FEATURE_DIM,
        output_dim=PROJ_DIM
    ).to(device)

    model = VICRegL_Uni(
        uni_model,
        projector
    ).to(device)

    print(
        f"[INFO] Loading checkpoint: "
        f"{CHECKPOINT}"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.projector.load_state_dict(
        checkpoint["projector"]
    )

    model.pe_linear.load_state_dict(
        checkpoint["pe_linear"]
    )

    return model, uni_preproc


def extract_features():
    df = pd.read_csv(
        CSV_FILE
    )

    print(
        f"[INFO] Total tiles: "
        f"{len(df)}"
    )

    model, uni_preproc = load_model()

    model.eval()

    dataset = TileDataset(
        df,
        uni_preproc
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, positions, paths in tqdm(
            loader,
            desc="Extracting"
        ):
            images = images.to(device)
            positions = positions.to(device)

            features = model(
                images,
                positions
            )

            all_features.append(
                features.cpu().numpy()
            )

            all_paths.extend(
                paths
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    all_paths = np.array(
        all_paths
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("[INFO] Saving features...")

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "tile_features_128.npy"
        ),
        features
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "tile_paths.npy"
        ),
        all_paths
    )

    print("[DONE] Features saved.")


if __name__ == "__main__":
    extract_features()