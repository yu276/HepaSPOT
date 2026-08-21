import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

from uni import get_encoder


DATA_CSV = "/path/to/tile_paths.csv"
D_IR_FILE = "/path/to/high_stable_tiles.pkl"
CHECKPOINT_DIR = "/path/to/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 30
IMG_SIZE = 224
FEATURE_DIM = 1536
PROJ_DIM = 128
LR = 1e-4
NUM_WORKERS = 8
ALPHA_SEM = 0.05
LSEM_MARGIN = 0.3

print(f"[INFO] Using device: {device}")


class TileDataset(Dataset):
    def __init__(self, tile_csv, uni_preproc=None):
        self.df = tile_csv
        self.uni_preproc = uni_preproc
        self.resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

        self.aug1 = transforms.Compose([
            self.resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2
            ),
            transforms.RandomGrayscale(p=0.2)
        ])

        self.aug2 = transforms.Compose([
            self.resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ColorJitter(
                brightness=0.6,
                contrast=0.6,
                saturation=0.6,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.2)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["tile_path"]

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        v1 = self.aug1(img)
        v2 = self.aug2(img)

        if self.uni_preproc:
            try:
                v1 = self.uni_preproc(v1)
                v2 = self.uni_preproc(v2)
            except Exception:
                return self.__getitem__((idx + 1) % len(self))

        return v1, v2, path


def vicregl_loss(
    z1,
    z2,
    sim_weight=25,
    std_weight=25,
    cov_weight=1
):
    batch_size, dim = z1.size()

    repr_loss = ((z1 - z2) ** 2).mean()

    std_z1 = z1.std(dim=0)
    std_z2 = z2.std(dim=0)

    std_loss = (
        torch.relu(1 - std_z1)
        + torch.relu(1 - std_z2)
    ).mean()

    z1_norm = z1 - z1.mean(dim=0)
    z2_norm = z2 - z2.mean(dim=0)

    c1 = (z1_norm.T @ z1_norm) / batch_size
    c2 = (z2_norm.T @ z2_norm) / batch_size

    def off_diag(x):
        return (
            x.flatten()[:-1]
            .view(dim - 1, dim + 1)[:, 1:]
            .flatten()
        )

    cov_loss = (
        off_diag(c1).pow(2).sum() / dim
        + off_diag(c2).pow(2).sum() / dim
    )

    loss = (
        sim_weight * repr_loss
        + std_weight * std_loss
        + cov_weight * cov_loss
    )

    return loss


class MemoryBank:
    def __init__(self, feature_dim, max_size=2048):
        self.feature_dim = feature_dim
        self.max_size = max_size
        self.bank = []

    def add(self, embeddings, labels):
        for emb, label in zip(
            embeddings.detach().cpu(),
            labels.cpu()
        ):
            self.bank.append({
                "emb": emb,
                "label": int(label)
            })

        if len(self.bank) > self.max_size:
            self.bank = self.bank[-self.max_size:]

    def sample(self):
        if len(self.bank) == 0:
            return None, None

        embeddings = torch.stack([
            item["emb"]
            for item in self.bank
        ])

        labels = torch.tensor(
            [
                item["label"]
                for item in self.bank
            ],
            dtype=torch.long
        )

        return embeddings, labels


def l_sem_memory_batch(
    batch_emb,
    batch_labels,
    mem_bank,
    margin=LSEM_MARGIN,
    device=device,
    alpha_pos=1.0,
    alpha_neg=1.0
):
    batch_size, _ = batch_emb.size()
    loss = 0.0

    same_mask = (
        batch_labels.unsqueeze(0)
        == batch_labels.unsqueeze(1)
    )

    diag_mask = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=device
    )

    same_mask = same_mask & (~diag_mask)

    if same_mask.sum() > 0:
        dist = 1 - torch.cosine_similarity(
            batch_emb.unsqueeze(1),
            batch_emb.unsqueeze(0),
            dim=2
        )

        loss_same = (
            dist * same_mask.float()
        ).sum() / same_mask.sum()

        loss += alpha_pos * loss_same

    mem_embs, mem_labels = mem_bank.sample()

    if mem_embs is not None:
        mem_embs = mem_embs.to(device)
        mem_labels = mem_labels.to(device)

        same_mask_mem = (
            batch_labels.unsqueeze(1)
            == mem_labels.unsqueeze(0)
        )

        if same_mask_mem.sum() > 0:
            dist_same_mem = 1 - torch.cosine_similarity(
                batch_emb.unsqueeze(1),
                mem_embs.unsqueeze(0),
                dim=2
            )

            loss_same_mem = (
                dist_same_mem
                * same_mask_mem.float()
            ).sum() / same_mask_mem.sum()

            loss += alpha_pos * loss_same_mem

        diff_mask_mem = ~same_mask_mem

        if diff_mask_mem.sum() > 0:
            dist_diff_mem = torch.cosine_similarity(
                batch_emb.unsqueeze(1),
                mem_embs.unsqueeze(0),
                dim=2
            )

            loss_diff_mem = torch.relu(
                dist_diff_mem - margin
            )

            loss_diff_mem = (
                loss_diff_mem
                * diff_mask_mem.float()
            ).sum() / diff_mask_mem.sum()

            loss += alpha_neg * loss_diff_mem

    return loss


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

    def forward(self, x):
        with torch.no_grad():
            feat = self.uni(x)

        proj = self.projector(feat)

        return proj


def train():
    df = pd.read_csv(DATA_CSV)

    with open(D_IR_FILE, "rb") as f:
        stable_tiles = pickle.load(f)

    stable_tile_dict = {
        item["tile"]: {
            "top1": item["top1"]
        }
        for item in stable_tiles
    }

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

    dataset = TileDataset(
        df,
        uni_preproc
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    optimizer = optim.Adam(
        model.projector.parameters(),
        lr=LR
    )

    memory_bank = MemoryBank(
        feature_dim=PROJ_DIM,
        max_size=2048
    )

    print("Starting training.")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            loader,
            desc=f"Epoch {epoch}"
        )

        for batch in progress_bar:
            v1, v2, paths = batch

            v1 = v1.to(device)
            v2 = v2.to(device)

            z1 = model(v1)
            z2 = model(v2)

            loss_vic = vicregl_loss(
                z1,
                z2
            )

            stable_embeddings = []
            stable_labels = []

            for path, embedding in zip(
                paths,
                z1
            ):
                if path in stable_tile_dict:
                    stable_embeddings.append(
                        embedding
                    )

                    stable_labels.append(
                        stable_tile_dict[path]["top1"]
                    )

            if len(stable_embeddings) > 0:
                stable_embeddings = torch.stack(
                    stable_embeddings
                )

                stable_labels = torch.tensor(
                    stable_labels,
                    dtype=torch.long,
                    device=device
                )

                loss_sem = l_sem_memory_batch(
                    stable_embeddings,
                    stable_labels,
                    memory_bank,
                    margin=LSEM_MARGIN,
                    device=device
                )

                memory_bank.add(
                    stable_embeddings,
                    stable_labels
                )

            else:
                loss_sem = torch.tensor(
                    0.0,
                    device=device
                )

            loss = (
                loss_vic
                + ALPHA_SEM * loss_sem
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                "Total_loss": f"{loss.item():.4f}",
                "VICRegL": f"{loss_vic.item():.4f}",
                "L_sem": f"{loss_sem.item():.4f}"
            })

        avg_loss = epoch_loss / len(loader)

        print(
            f"[Epoch {epoch}] "
            f"Average total loss: {avg_loss:.4f}"
        )

        checkpoint_file = os.path.join(
            CHECKPOINT_DIR,
            f"epoch_{epoch:03d}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "projector": model.projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss": avg_loss
            },
            checkpoint_file
        )

        print(
            f"Checkpoint saved: {checkpoint_file}"
        )


if __name__ == "__main__":
    train()