# If needed:
# !pip install -q transformers accelerate safetensors

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import AutoModel, AutoImageProcessor
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

IMAGES_DIR = "/scratch/sejong/class-dataset/cxr-256/nih_xray14/images/images"

CSV_TRAIN = "/scratch/sejong/class-dataset/cxr-ground-truth/xray14/train_official.csv"
CSV_VAL   = "/scratch/sejong/class-dataset/cxr-ground-truth/xray14/val_official.csv"
CSV_TEST  = "/scratch/sejong/class-dataset/cxr-ground-truth/xray14/test_official.csv"

repo = "microsoft/rad-dino"

processor = AutoImageProcessor.from_pretrained(repo)

# 🔥 ADD THIS RIGHT HERE
processor = AutoImageProcessor.from_pretrained(repo)

# 🔥 FORCE RESIZE
processor.size = {"shortest_edge": 256}
processor.crop_size = {"height": 256, "width": 256}



import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ChestXray14Dataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor
        self.labels = df[LABELS].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = row["Path"]

        # 🔥 FIX: handle NIH path correctly
        if "images/images" in path:
            img_path = os.path.join("/scratch/sejong/class-dataset/cxr-256/nih_xray14", path)
        else:
            img_path = os.path.join(self.img_dir, path)

        image = Image.open(img_path).convert("RGB")

        # 🔥 RAD-DINO processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return pixel_values, label


import pandas as pd

train_df = pd.read_csv(CSV_TRAIN)
val_df   = pd.read_csv(CSV_VAL)
test_df  = pd.read_csv(CSV_TEST)

train_ds = ChestXray14Dataset(train_df, IMAGES_DIR, processor)
val_ds   = ChestXray14Dataset(val_df, IMAGES_DIR, processor)
test_ds  = ChestXray14Dataset(test_df, IMAGES_DIR, processor)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

def unfreeze_last_layers(model, num_layers=10):
    layers = model.encoder.encoder.layer  

    total_layers = len(layers)
    print(f"Total transformer layers: {total_layers}")
    print(f"Unfreezing last {num_layers} layers")

    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True




import torch
import torch.nn as nn

import torch
import torch.nn as nn

class RadDinoClassifier(nn.Module):
    def __init__(self, encoder, num_classes=14, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

        # 🔥 Improved MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # 🔒 Freeze encoder initially
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)

        # 🔥 CRITICAL FIX (this is why performance improves)
        feat = outputs.last_hidden_state[:, 1:, :]  # remove CLS token
        feat = feat.mean(dim=1)                     # global pooling

        logits = self.head(feat)
        return logits


import torch
import torch.nn as nn

class RadDinoMLP(nn.Module):
    def __init__(self, encoder, num_classes=14, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

        # 🔥 Improved head (no over-compression)
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)

        # 🔥 CRITICAL FIX
        feat = outputs.last_hidden_state[:, 1:, :]   # remove CLS token
        feat = feat.mean(dim=1)                      # global average pooling

        logits = self.head(feat)
        return logits

model = RadDinoMLP(rad_dino, num_classes=14, freeze_encoder=True).to(device)

def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)   # ✅ FIX
            labels = labels.to(device)

            logits = model(pixel_values)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc_scores = []
    for i in range(len(LABELS)):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except ValueError:
            auc = float("nan")
        auc_scores.append(auc)

    macro_auc = np.nanmean(auc_scores)
    return macro_auc, auc_scores


from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for pixel_values, labels in pbar:   # ✅ correct unpack
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits = model(pixel_values)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(n, 1)

encoder_lr = 3e-6   # 🔥 lower than before
head_lr = 5e-5

import torch

def compute_pos_weights(loader, device):
    total = None
    count = 0

    for images, labels in loader:
        labels = labels.to(device)

        if total is None:
            total = torch.zeros(labels.shape[1]).to(device)

        total += labels.sum(dim=0)
        count += labels.size(0)

    pos_counts = total
    neg_counts = count - pos_counts

    pos_weights = neg_counts / (pos_counts + 1e-6)

    return pos_weights


# 🔥 Compute weights
pos_weights = compute_pos_weights(train_loader, device)

print("Class weights:", pos_weights)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# Normalize (IMPORTANT)
pos_weights = pos_weights / pos_weights.mean()

# OPTIONAL: boost weak classes
INFILTRATION_IDX = LABELS.index("Infiltration")
PNEUMONIA_IDX = LABELS.index("Pneumonia")
NODULE_IDX = LABELS.index("Nodule")

pos_weights[INFILTRATION_IDX] *= 1.3
pos_weights[PNEUMONIA_IDX] *= 1.3
pos_weights[NODULE_IDX] *= 1.2

print("🔥 Final class weights:", pos_weights)


# =========================
# 🔥 LOSS
# =========================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)


# =========================
# 🔥 OPTIMIZER (LOW LR)
# =========================
encoder_params = []
head_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if "encoder" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": 3e-6},   # 🔥 CRITICAL
    {"params": head_params, "lr": 5e-5}
], weight_decay=1e-4)


# =========================
# 🔥 TRAINING LOOP
# =========================
EPOCHS = 15
best_auc = -1
patience = 2
no_improve = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_auc, val_class_auc = evaluate(model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val AUC: {val_auc:.4f}")

    # Per-class AUC
    for label, auc in zip(LABELS, val_class_auc):
        if auc is not None:
            print(f"{label:30s}: {auc:.4f}")

    # Save best model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "best_rad_dino_unfreeze10.pth")
        print("✅ Model saved!")
        no_improve = 0
    else:
        no_improve += 1

    # Early stopping
    if no_improve >= patience:
        print("⛔ Early stopping triggered")
        break


model.load_state_dict(torch.load("best_rad_dino_unfreeze10.pth"))

test_auc, class_auc = evaluate(model, test_loader, device)

print("\n🔥 FINAL TEST RESULTS")
print(f"Macro AUC: {test_auc:.4f}")

for i, cls in enumerate(LABELS):
    print(f"{cls:20}: {class_auc[i]:.4f}")
