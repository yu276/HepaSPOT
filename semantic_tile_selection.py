# -*- coding: utf-8 -*-

import os
import json
import pickle
import random

import torch
import pandas as pd
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm


device = "cuda:0" if torch.cuda.is_available() else "cpu"


model_path = "/path/to/plip_model"

model_plip = CLIPModel.from_pretrained(
    model_path
).to(device).eval()

plip_processor = CLIPProcessor.from_pretrained(
    model_path
)


prompt_list = [
    "histopathology image of hepatocellular carcinoma tumor cells",
    "cholangiocarcinoma histopathology tissue",
    "liver fibrosis cirrhotic stroma histopathology",
    "dense inflammatory cell infiltration in liver tissue",
    "hepatic sinusoidal vascular structures histology",
    "tissue necrosis in liver histopathology",
    "hepatic steatosis fat droplets histology",
    "normal liver parenchyma histology"
]


with torch.no_grad():

    inputs_text = plip_processor(
        text=prompt_list,
        return_tensors="pt",
        padding=True
    ).to(device)

    text_emb = model_plip.get_text_features(
        **inputs_text
    )

    text_emb = (
        text_emb
        / text_emb.norm(dim=-1, keepdim=True)
    )


data_csv = "/path/to/tile_paths.csv"

data_df = pd.read_csv(data_csv)

tile_paths = data_df["tile_path"].tolist()


def augment_image(img):

    img = F.resize(
        img,
        (224, 224)
    )

    angle = random.choice(
        [0, 90, 180, 270]
    )

    img = F.rotate(
        img,
        angle
    )

    img = transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    )(img)

    img = transforms.ToTensor()(img)

    return img


K = 8

tile_features = []


for tile_path in tqdm(
    tile_paths,
    desc="Processing tiles"
):

    try:
        img = Image.open(
            tile_path
        ).convert("RGB")

    except Exception as e:

        print(
            f"Warning: unable to open "
            f"{tile_path}: {e}"
        )

        continue


    all_p = []


    for _ in range(K):

        img_aug = augment_image(
            img
        )

        inputs_img = plip_processor(
            images=img_aug,
            return_tensors="pt",
            do_rescale=False
        ).to(device)


        with torch.no_grad():

            image_emb = (
                model_plip.get_image_features(
                    **inputs_img
                )
            )


        image_emb = (
            image_emb
            / image_emb.norm(
                dim=-1,
                keepdim=True
            )
        )


        similarity = (
            image_emb
            @ text_emb.T
        )


        probabilities = torch.softmax(
            similarity,
            dim=-1
        )


        all_p.append(
            probabilities.squeeze(0)
            .detach()
            .cpu()
        )


    p_avg = (
        torch.stack(all_p)
        .mean(0)
    )


    entropy = -(
        p_avg
        * torch.log(
            p_avg + 1e-6
        )
    ).sum().item()


    top1 = torch.argmax(
        p_avg
    ).item()


    tile_features.append(
        {
            "tile": tile_path,
            "p_avg": p_avg.numpy().tolist(),
            "entropy": entropy,
            "top1": top1
        }
    )


all_entropy = np.array(
    [
        tile["entropy"]
        for tile in tile_features
    ]
)


threshold = np.quantile(
    all_entropy,
    0.20
)


high_stable_tiles = []
unstable_tiles = []


for tile in tile_features:

    if tile["entropy"] <= threshold:

        high_stable_tiles.append(
            tile
        )

    else:

        unstable_tiles.append(
            tile
        )


print(
    f"High-stability tiles: "
    f"{len(high_stable_tiles)}"
)

print(
    f"Unstable tiles: "
    f"{len(unstable_tiles)}"
)


output_dir = "/path/to/output_directory"

os.makedirs(
    output_dir,
    exist_ok=True
)


with open(
    os.path.join(
        output_dir,
        "high_stable_tiles.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        high_stable_tiles,
        f
    )


with open(
    os.path.join(
        output_dir,
        "unstable_tiles.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        unstable_tiles,
        f
    )


with open(
    os.path.join(
        output_dir,
        "tile_mvc_features.json"
    ),
    "w"
) as f:

    json.dump(
        tile_features,
        f
    )


print(
    "Semantic tile stability assessment completed."
)