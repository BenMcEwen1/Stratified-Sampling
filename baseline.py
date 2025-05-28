import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import os
import pickle
from pathlib import Path

dataset_path = "./WABAD"

BIRDNET_LABELS = Path(os.path.join(dataset_path,"data_files"), "BirdNET_GLOBAL_6K_V2.4_Labels.txt").read_text(encoding="utf-8").splitlines()
BIRDNET_LATIN_LABELS = [item.split('_')[0] for item in BIRDNET_LABELS]
BIRDNET_ENGLISH_LABELS = [item.split('_')[1] for item in BIRDNET_LABELS]

# Species that occure in at least one 1-minute files in both train and test sets (curruca cantillans removed)
bird_list = ['Acrocephalus arundinaceus', 'Acrocephalus scirpaceus', 'Alauda arvensis', 'Alectoris chukar', 'Anas platyrhynchos', 'Anthus pratensis', 'Anthus trivialis', 'Ardea cinerea', 'Athene noctua', 'Burhinus oedicnemus', 'Calandrella brachydactyla', 'Carduelis carduelis', 'Certhia brachydactyla', 'Cettia cetti', 'Chloris chloris', 'Chroicocephalus ridibundus', 'Circus aeruginosus', 'Cisticola juncidis', 'Coccothraustes coccothraustes', 'Columba oenas', 'Columba palumbus', 'Corvus corax', 'Corvus cornix', 'Corvus corone', 'Corvus monedula', 'Cuculus canorus', 'Curruca communis', 'Curruca conspicillata', 'Curruca iberiae', 'Curruca melanocephala', 'Curruca undata', 'Cyanistes caeruleus', 'Cyanopica cooki', 'Dendrocopos major', 'Dendrocoptes medius', 'Dryocopus martius', 'Emberiza calandra', 'Emberiza cia', 'Emberiza cirlus', 'Emberiza citrinella', 'Erithacus rubecula', 'Falco tinnunculus', 'Ficedula albicollis', 'Fringilla coelebs', 'Fringilla montifringilla', 'Fulica atra', 'Galerida cristata', 'Galerida theklae', 'Gallinula chloropus', 'Gallus gallus', 'Garrulus glandarius', 'Grus grus', 'Himantopus himantopus', 'Hippolais polyglotta', 'Lanius senator', 'Larus argentatus', 'Linaria cannabina', 'Locustella naevia', 'Lophophanes cristatus', 'Lullula arborea', 'Luscinia megarhynchos', 'Merops apiaster', 'Milvus migrans', 'Motacilla flava', 'Oriolus oriolus', 'Parus major', 'Passer domesticus', 'Periparus ater', 'Petronia petronia', 'Phasianus colchicus', 'Phoenicopterus roseus', 'Phoenicurus ochruros', 'Phylloscopus bonelli', 'Phylloscopus collybita', 'Phylloscopus sibilatrix', 'Phylloscopus trochilus', 'Pica pica', 'Picus sharpei', 'Poecile palustris', 'Prunella collaris', 'Prunella modularis', 'Pyrrhocorax pyrrhocorax', 'Pyrrhula pyrrhula', 'Rallus aquaticus', 'Recurvirostra avosetta', 'Regulus ignicapilla', 'Regulus regulus', 'Remiz pendulinus', 'Saxicola rubetra', 'Saxicola rubicola', 'Serinus serinus', 'Sitta europaea', 'Spinus spinus', 'Streptopelia decaocto', 'Sturnus unicolor', 'Sylvia atricapilla', 'Tachybaptus ruficollis', 'Tadorna ferruginea', 'Tadorna tadorna', 'Tringa glareola', 'Tringa ochropus', 'Troglodytes troglodytes', 'Turdus merula', 'Turdus philomelos', 'Turdus viscivorus', 'Upupa epops']
birds_list_index = []
for specie_name in bird_list:
    birds_list_index.append( BIRDNET_LATIN_LABELS.index(specie_name))
bird_list_english = (np.asarray(BIRDNET_ENGLISH_LABELS)[birds_list_index]).tolist()
labels_name = bird_list

# load train labels and sampling data
with open(os.path.join(dataset_path,"data_files" ,  "dataframe_train.pkl"), "rb") as file:
    loaded_data = pickle.load(file)

[samples_df_train, y_true_train, y_scores_train] = loaded_data

train_sets_size = y_true_train.shape[0]
y_true_train = y_true_train.astype(int)
y_true_train = y_true_train[:,birds_list_index]

birdnet_scores_train = y_scores_train[:,birds_list_index]

# load validation labels
with open(os.path.join(dataset_path,"data_files" , "dataframe_validation.pkl"), "rb") as file:
    loaded_data = pickle.load(file)
[samples_df_val, y_true_val, y_scores_val] = loaded_data

train_val_size = y_true_val.shape[0]
y_true_val = y_true_val.astype(int)
y_true_val = y_true_val[:,birds_list_index]

# load test embeddings
with open(os.path.join(dataset_path,"data_files" , "embeddings_train.pkl"), "rb") as file:
    loaded_data = pickle.load(file)
[embeddings_train, files_list] = loaded_data

verif = (samples_df_train["File"] == files_list).tolist()

# load validation embeddings
with open(os.path.join(dataset_path,"data_files" , "embeddings_validation.pkl"), "rb") as file:
    loaded_data = pickle.load(file)
[embeddings_val, files_list] = loaded_data

verif = (samples_df_val["File"] == files_list).tolist()

# Select species in the bird list
num_classes =  len(labels_name)

## Train - validation split
x_train = embeddings_train
x_val = embeddings_val
y_train = y_true_train
y_val = y_true_val

# # Needs for resampling
full_train = x_train 
full_labels = y_train
print('Full training set', np.shape(full_train))


tensor_x_train = torch.Tensor(full_train)
tensor_y_train = torch.Tensor(full_labels)
# tensor_x_train = torch.Tensor(x_train) # transform to torch tensor
# tensor_y_train = torch.Tensor(y_train)
tensor_x_val = torch.Tensor(x_val) # transform to torch tensor
tensor_y_val = torch.Tensor(y_val)

# val_dataset = TensorDataset(tensor_x_val, tensor_y_val) # create your datset
# val_loader = DataLoader(val_dataset, batch_size=batch_size) # create your dataloader

train_filenames = samples_df_train["File"]

#load test labels
with open(os.path.join(dataset_path,"data_files" , "dataframe_test.pkl"), "rb") as file:
    loaded_data = pickle.load(file)

[samples_df_test, y_true_test, y_scores_test] = loaded_data

y_true_test = y_true_test[:,birds_list_index]
y_true_test = y_true_test.astype(int)

num_samples_test = np.shape(y_true_test)[0]
species_count_test = np.sum(y_true_test,axis=0)

#Load test embeddings
with open(os.path.join(dataset_path,"data_files" , "embeddings_test.pkl"), "rb") as file:
    loaded_data = pickle.load(file)

[embeddings_test, files_list_test] = loaded_data
verif = (samples_df_test["File"] == files_list_test).tolist()

tensor_x_test = torch.tensor(embeddings_test)
tensor_y_test = torch.tensor(y_true_test)

test_filenames = samples_df_test["File"]


from adapter import loader

DATASET_PATH = "anuraset"

tensor_x_train, tensor_y_train, train_filenames = loader(DATASET_PATH, sub_directory="train")
tensor_x_test, tensor_y_test, test_filenames = loader(DATASET_PATH, sub_directory="test")

# print(tensor_y_train.sum(axis=0))
common = tensor_y_train.sum(axis=0) > 5000
rare = tensor_y_train.sum(axis=0) < 5000
all_classes = tensor_y_train.sum(axis=0) > 0
filters = all_classes

train_embeddings = tensor_x_train
train_labels = tensor_y_train[:, filters]

val_embeddings = tensor_x_test
val_labels = tensor_y_test[:, filters]

test_embeddings = tensor_x_test
test_labels = tensor_y_test[:, filters]


from torch.optim.lr_scheduler import ReduceLROnPlateau


# --- Config ---
NUM_CLASSES = train_labels.shape[1]
INIT_SIZE = 100
BATCH_SIZE = 32
AL_EPOCHS = 25
QUERY_SIZE = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Simple classifier ---
class Classifier(nn.Module):
    def __init__(self, in_dim=1024, out_dim=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

# --- Binary entropy scorer ---
def binary_entropy(logits):
    probs = torch.sigmoid(logits)
    entropy = - (probs * torch.log(probs + 1e-10) + (1 - probs) * torch.log(1 - probs + 1e-10))
    per_sample_entropy = torch.max(entropy, axis=1)[0] 
    return per_sample_entropy

# --- Training loop ---
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.float(), y.float())
        loss.backward()
        optimizer.step()

# --- Eval (mAP) ---
def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(X.to(DEVICE))
        probs = torch.sigmoid(logits).cpu()
    # mAP: per-class average precision
    from sklearn.metrics import average_precision_score, f1_score

    valid_classes = Y.sum(axis=0) > 0
    Y = Y[:, valid_classes]
    probs = probs[:, valid_classes]
    preds = (probs > 0.5).float()

    mAP = average_precision_score(Y, probs, average="weighted", sample_weight=None)
    cmAP = average_precision_score(Y, probs, average="macro", sample_weight=None)
    F1_macro = f1_score(Y, preds, average="macro")
    return mAP, cmAP, F1_macro

# --- Active Learning loop ---
def active_learning_loop(strategy='random'):
    results = []
    model = Classifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=0.001)

    unlabeled_idx = np.arange(len(train_embeddings))
    labeled_idx = np.random.choice(unlabeled_idx, size=INIT_SIZE, replace=False).tolist()
    unlabeled_idx = list(set(unlabeled_idx) - set(labeled_idx))

    for epoch in range(AL_EPOCHS):
        # Train on current labeled set
        train_loader = DataLoader(Subset(TensorDataset(train_embeddings, train_labels), labeled_idx),
                                  batch_size=BATCH_SIZE, shuffle=True)
        train_model(model, train_loader, optimizer, criterion)

        # Evaluate
        mAP, cmAP, F1_macro = evaluate(model, val_embeddings, val_labels)
        print(f"[{strategy}] Epoch {epoch+1} | mAP: {mAP:.4f} cmAP: {cmAP:.4f} | F1 Macro: {F1_macro:.4f}  | Labeled: {len(labeled_idx)}")
        results.append(F1_macro)
        scheduler.step(F1_macro)

        if len(unlabeled_idx) == 0:
            break  # no more to query

        # Select new points to label
        model.eval()
        with torch.no_grad():
            logits = model(train_embeddings[unlabeled_idx].to(DEVICE))
            if strategy == 'random':
                scores = torch.rand(len(unlabeled_idx))
            elif strategy == 'binary':
                scores = binary_entropy(logits)
            else:
                raise ValueError("Unknown strategy")

        topk = torch.topk(scores, k=min(QUERY_SIZE, len(unlabeled_idx))).indices.cpu().numpy()
        new_indices = [unlabeled_idx[i] for i in topk]

        # Accumulate
        labeled_idx.extend(new_indices)
        unlabeled_idx = [i for i in unlabeled_idx if i not in new_indices]

    # Final test set performance
    mAP, cmAP, F1_macro = evaluate(model, test_embeddings, test_labels)
    print(f"[{strategy}] Final Test mAP: {mAP:.4f}")
    return results

mAP_random = active_learning_loop(strategy='random')
mAP_binary = active_learning_loop(strategy='binary')

import matplotlib.pyplot as plt

plt.plot(mAP_random, label='random')
plt.plot(mAP_binary, label='binary')
plt.legend()
plt.show()