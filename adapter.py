"""
Middleware script for running active learning pipeline on datasets
Load:
- Training set - embeddings, labels
- Validation set (optional) - embeddings, labels
- Test set - embeddings, labels
- Spatial strata - list of locations (ordered by sample indice)
- Temporal strata - list of dates (ordered by sample indices)

Supported datasets:
- AnuraSet: Standard format with embeddings.pt, labels.pt, and data.csv
- WABAD: Custom format with pickle files for embeddings and dataframes
"""

import pandas as pd
import torch
import pickle
import os
import numpy as np
from pathlib import Path


def loader(dataset_name, sub_directory="train", dataset_path=None):
    """
    Load dataset embeddings, labels, and filenames for different datasets.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ("anuraset" or "wabad")
    sub_directory : str
        Subdirectory to load from ("train", "test", or "validation")
    dataset_path : str, optional
        Custom path to dataset. If None, uses default paths.

    Returns:
    --------
    embeddings : torch.Tensor
        Embedding vectors
    labels : torch.Tensor
        Label vectors
    filenames : list
        List of filenames
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "anuraset":
        return _load_anuraset(dataset_path or "anuraset", sub_directory)
    elif dataset_name == "wabad":
        return _load_wabad(dataset_path or "./WABAD", sub_directory)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported datasets: 'anuraset', 'wabad'")


def _load_anuraset(dataset_path, sub_directory):
    """Load AnuraSet dataset in standard format."""
    embeddings = torch.load(f"{dataset_path}/{sub_directory}/embeddings.pt", weights_only=True)
    labels = torch.load(f"{dataset_path}/{sub_directory}/labels.pt", weights_only=True)

    df = pd.read_csv(f"{dataset_path}/{sub_directory}/data.csv")
    filenames = df['fname'].to_list()
    return embeddings, labels, filenames


def _load_wabad(dataset_path, sub_directory):
    """Load WABAD dataset from pickle files."""
    # Map sub_directory to WABAD naming convention
    dir_map = {
        "train": "train",
        "test": "test",
        "validation": "validation",
        "val": "validation"
    }

    wabad_dir = dir_map.get(sub_directory, sub_directory)
    data_files_path = os.path.join(dataset_path, "data_files")

    # Load dataframe and scores
    dataframe_file = os.path.join(data_files_path, f"dataframe_{wabad_dir}.pkl")
    with open(dataframe_file, "rb") as file:
        loaded_data = pickle.load(file)

    samples_df, y_true, y_scores = loaded_data

    # Load embeddings
    embeddings_file = os.path.join(data_files_path, f"embeddings_{wabad_dir}.pkl")
    with open(embeddings_file, "rb") as file:
        loaded_data = pickle.load(file)

    embeddings, files_list = loaded_data

    # Load bird list labels (106 species that occur in both train and test)
    BIRDNET_LABELS = Path(os.path.join(dataset_path, "data_files"), "BirdNET_GLOBAL_6K_V2.4_Labels.txt").read_text(encoding="utf-8").splitlines()
    BIRDNET_LATIN_LABELS = [item.split('_')[0] for item in BIRDNET_LABELS]

    bird_list = ['Acrocephalus arundinaceus', 'Acrocephalus scirpaceus', 'Alauda arvensis', 'Alectoris chukar', 'Anas platyrhynchos', 'Anthus pratensis', 'Anthus trivialis', 'Ardea cinerea', 'Athene noctua', 'Burhinus oedicnemus', 'Calandrella brachydactyla', 'Carduelis carduelis', 'Certhia brachydactyla', 'Cettia cetti', 'Chloris chloris', 'Chroicocephalus ridibundus', 'Circus aeruginosus', 'Cisticola juncidis', 'Coccothraustes coccothraustes', 'Columba oenas', 'Columba palumbus', 'Corvus corax', 'Corvus cornix', 'Corvus corone', 'Corvus monedula', 'Cuculus canorus', 'Curruca communis', 'Curruca conspicillata', 'Curruca iberiae', 'Curruca melanocephala', 'Curruca undata', 'Cyanistes caeruleus', 'Cyanopica cooki', 'Dendrocopos major', 'Dendrocoptes medius', 'Dryocopus martius', 'Emberiza calandra', 'Emberiza cia', 'Emberiza cirlus', 'Emberiza citrinella', 'Erithacus rubecula', 'Falco tinnunculus', 'Ficedula albicollis', 'Fringilla coelebs', 'Fringilla montifringilla', 'Fulica atra', 'Galerida cristata', 'Galerida theklae', 'Gallinula chloropus', 'Gallus gallus', 'Garrulus glandarius', 'Grus grus', 'Himantopus himantopus', 'Hippolais polyglotta', 'Lanius senator', 'Larus argentatus', 'Linaria cannabina', 'Locustella naevia', 'Lophophanes cristatus', 'Lullula arborea', 'Luscinia megarhynchos', 'Merops apiaster', 'Milvus migrans', 'Motacilla flava', 'Oriolus oriolus', 'Parus major', 'Passer domesticus', 'Periparus ater', 'Petronia petronia', 'Phasianus colchicus', 'Phoenicopterus roseus', 'Phoenicurus ochruros', 'Phylloscopus bonelli', 'Phylloscopus collybita', 'Phylloscopus sibilatrix', 'Phylloscopus trochilus', 'Pica pica', 'Picus sharpei', 'Poecile palustris', 'Prunella collaris', 'Prunella modularis', 'Pyrrhocorax pyrrhocorax', 'Pyrrhula pyrrhula', 'Rallus aquaticus', 'Recurvirostra avosetta', 'Regulus ignicapilla', 'Regulus regulus', 'Remiz pendulinus', 'Saxicola rubetra', 'Saxicola rubicola', 'Serinus serinus', 'Sitta europaea', 'Spinus spinus', 'Streptopelia decaocto', 'Sturnus unicolor', 'Sylvia atricapilla', 'Tachybaptus ruficollis', 'Tadorna ferruginea', 'Tadorna tadorna', 'Tringa glareola', 'Tringa ochropus', 'Troglodytes troglodytes', 'Turdus merula', 'Turdus philomelos', 'Turdus viscivorus', 'Upupa epops']

    birds_list_index = []
    for specie_name in bird_list:
        birds_list_index.append(BIRDNET_LATIN_LABELS.index(specie_name))

    # Filter to bird list species
    y_true = y_true.astype(int)
    y_true = y_true[:, birds_list_index]

    # Convert to tensors
    tensor_embeddings = torch.Tensor(embeddings)
    tensor_labels = torch.Tensor(y_true)
    filenames = samples_df["File"].tolist()

    return tensor_embeddings, tensor_labels, filenames


