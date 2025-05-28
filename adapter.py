"""
Middleware script for running active learning pipeline on datasets
Load:
- Training set - embeddings, labels
- Validation set (optional) - embeddings, labels
- Test set - embeddings, labels
- Spatial strata - list of locations (ordered by sample indice)
- Temporal strata - list of dates (ordered by sample indices)
"""

"""
Dataset structure:
train_x = torch.load("train/embeddings.pt")
train_y = torch.load("train/labels.pt")
train_df = pd.read_csv("train/train_set.csv")
"""

import pandas as pd
import torch

DATASET_PATH = "anuraset"

def loader(DATASET_PATH, sub_directory="train"):
    embeddings = torch.load(f"{DATASET_PATH}/{sub_directory}/embeddings.pt", weights_only=True)
    labels = torch.load(f"{DATASET_PATH}/{sub_directory}/labels.pt", weights_only=True)

    df = pd.read_csv(f"{DATASET_PATH}/{sub_directory}/data.csv")
    filenames = df['fname'].to_list()
    return embeddings, labels, filenames


