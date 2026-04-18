!pip install -q transformers timm
import os
import torch
import timm
import pandas as pd
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoModel, AutoImageProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

repo = "microsoft/rad-dino"

# Processor
processor = AutoImageProcessor.from_pretrained(repo)
processor.size = {"shortest_edge": 256}
processor.crop_size = {"height": 256, "width": 256}

# Model
rad_dino = AutoModel.from_pretrained(repo)
rad_dino = rad_dino.to(device)

print("✅ Loaded RAD-DINO on", device)

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

class_to_idx = {cls: i for i, cls in enumerate(class_names)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

NUM_CLASSES = len(class_names)
import os

BASE_PATH = "/content/drive/MyDrive/datasets/HAM10000"

IMG_DIR = os.path.join(BASE_PATH, "HAM10000_images_part_1")  # + part_2 handled later
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
VAL_CSV   = os.path.join(BASE_PATH, "val.csv")

class HAMDataset(Dataset):
    def __init__(self, df, img_dir, processor):
        self.df = df
        self.img_dir = img_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row['image_id'] + ".jpg")
        image = Image.open(img_path).convert("RGB")

        # ✅ HuggingFace processor usage
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        label = class_to_idx[row['dx']]
        target = torch.zeros(NUM_CLASSES)
        target[label] = 1.0

        return pixel_values, target

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

from sklearn.model_selection import train_test_split
import pandas as pd
import os

BASE_PATH = "/content/drive/MyDrive/datasets/HAM10000"

df = pd.read_csv(os.path.join(BASE_PATH, "HAM10000_metadata.csv"))

# 🔥 Step 1: Train (70) + temp (30)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["dx"],
    random_state=42
)

# 🔥 Step 2: Val (20) + Test (10)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.3333,  # 10/30 = 0.333
    stratify=temp_df["dx"],
    random_state=42
)

print(len(train_df), len(val_df), len(test_df))
train_df.to_csv(os.path.join(BASE_PATH, "train.csv"), index=False)
val_df.to_csv(os.path.join(BASE_PATH, "val.csv"), index=False)
test_df.to_csv(os.path.join(BASE_PATH, "test.csv"), index=False)

print("✅ Splits saved")

train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)



def unfreeze_last_layers(model, num_layers=10):
    layers = model.encoder.encoder.layer  # ✅ confirmed from your print

    total_layers = len(layers)
    print(f"Total transformer layers: {total_layers}")
    print(f"Unfreezing last {num_layers} layers")

    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


from transformers import AutoImageProcessor

repo = "microsoft/rad-dino"

processor = AutoImageProcessor.from_pretrained(repo)

processor.size = {"shortest_edge": 256}
processor.crop_size = {"height": 256, "width": 256}

print(type(processor))  # ✅ should NOT be Compose

class RadDinoClassifier(nn.Module):
    def __init__(self, encoder, num_classes=7, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

        self.attn_pool = nn.Sequential(
            nn.Linear(768, 1),
            nn.Softmax(dim=1)
        )

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
        hidden = outputs.last_hidden_state

        feat_tokens = hidden[:, 1:, :]

        weights = self.attn_pool(feat_tokens)
        feat = (feat_tokens * weights).sum(dim=1)

        return self.head(feat)
import torch
import torch.nn as nn

class RadDinoMLP(nn.Module):
    def __init__(self, encoder, num_classes=7, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder

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

        hidden = outputs.last_hidden_state  # (B, N, 768)

        # 🔥 Remove CLS + mean pooling
        feat = hidden[:, 1:, :]
        feat = feat.mean(dim=1)

        logits = self.head(feat)
        return logits


NUM_CLASSES = 7  # 🔥 HAM10000

model = RadDinoMLP(
    rad_dino,
    num_classes=NUM_CLASSES,
    freeze_encoder=True
).to(DEVICE)

import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            logits = model(pixel_values)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu())
            all_targets.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    auc_scores = []
    for i in range(NUM_CLASSES):
        try:
            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except:
            auc = 0.5  # safer fallback than NaN
        auc_scores.append(auc)

    macro_auc = sum(auc_scores) / len(auc_scores)

    return macro_auc, auc_scores


def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for pixel_values, labels in pbar:
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

class HAMDataset(Dataset):
    def __init__(self, df, image_map, processor):
        assert isinstance(image_map, dict), f"❌ image_map must be dict, got {type(image_map)}"
        self.df = df.reset_index(drop=True)
        self.image_map = image_map
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ✅ Correct image lookup
        img_path = self.image_map[row["image_id"]]
        image = Image.open(img_path).convert("RGB")

        # ✅ HF processor (correct)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # ✅ One-hot target
        label = class_to_idx[row["dx"]]
        target = torch.zeros(NUM_CLASSES)
        target[label] = 1.0

        return pixel_values, target

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["dx"].map(class_to_idx)
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = torch.amp.GradScaler("cuda")

def compute_pos_weights(loader, device):
    total = None
    count = 0

    for _, labels in loader:
        labels = labels.to(device)

        if total is None:
            total = torch.zeros(labels.shape[1]).to(device)

        total += labels.sum(dim=0)
        count += labels.shape[0]

    pos_weights = (count - total) / (total + 1e-6)
    return pos_weights
pos_weights = compute_pos_weights(train_loader, device)

# normalize
pos_weights = pos_weights / pos_weights.mean()

print("Base pos_weights:", pos_weights)

# 🔥 TARGETED BOOSTING (your classes)
pos_weights[4] *= 1.5   # weakest
pos_weights[2] *= 1.3
pos_weights[3] *= 1.3
pos_weights[6] *= 1.2

# 🔥 clamp
pos_weights = torch.clamp(pos_weights, 0.5, 5.0)

print("🔥 Boosted weights:", pos_weights)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)


# =========================
# 🔥 OPTIMIZER (LOW LR)
# =========================
optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 5e-7},
    {"params": model.head.parameters(), "lr": 5e-6}
], weight_decay=1e-4)


# =========================
# 🔥 SCHEDULER
# =========================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)


# =========================
# 🔥 TRAINING LOOP
# =========================
EPOCHS = 10
best_auc = 0.0
patience = 3
no_improve = 0

print("\n🚀 Stage 2: 10-layer + boosted training")

for epoch in range(EPOCHS):

    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    # =========================
    # 🔥 TRAIN
    # =========================
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    train_loss = total_loss / len(train_loader)


    # =========================
    # 🔥 VALIDATION
    # =========================
    val_auc, val_class_auc = evaluate(model, val_loader, device)

    scheduler.step(val_auc)

    print(f"📉 Train Loss: {train_loss:.4f}")
    print(f"📊 Val AUC: {val_auc:.4f}")
    print(f"📊 Per-class AUC: {val_class_auc}")

    # =========================
    # 🔥 SAVE BEST MODEL
    # =========================
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved! AUC: {best_auc:.4f}")
        no_improve = 0
    else:
        no_improve += 1
        print(f"⚠️ No improvement ({no_improve}/{patience})")

    # =========================
    # 🔥 EARLY STOPPING
    # =========================
    if no_improve >= patience:
        print("🛑 Early stopping triggered")
        break

EPOCHS = 5   # 🔥 extend by 5 more epochs
patience = 4
no_improve = 0

print("\n🚀 Continuing training (fine-tuning phase)")

for epoch in range(EPOCHS):

    print(f"\n🔥 Continue Epoch {epoch+1}/{EPOCHS}")

    # =========================
    # 🔥 TRAIN
    # =========================
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    train_loss = total_loss / len(train_loader)


    # =========================
    # 🔥 VALIDATION
    # =========================
    val_auc, val_class_auc = evaluate(model, val_loader, device)

    scheduler.step(val_auc)

    print(f"📉 Train Loss: {train_loss:.4f}")
    print(f"📊 Val AUC: {val_auc:.4f}")
    print(f"📊 Per-class AUC: {val_class_auc}")


    # =========================
    # 🔥 SAVE BEST MODEL
    # =========================
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Improved model saved! AUC: {best_auc:.4f}")
        no_improve = 0
    else:
        no_improve += 1
        print(f"⚠️ No improvement ({no_improve}/{patience})")


    # =========================
    # 🔥 EARLY STOPPING
    # =========================
    if no_improve >= patience:
        print("🛑 Early stopping triggered")
        break

test_auc, test_class_auc = evaluate(model, test_loader, device)

print("\n🔥 FINAL TEST RESULTS")
print(f"Macro AUC: {test_auc:.4f}")

for i, auc in enumerate(test_class_auc):
    print(f"Class {i}: {auc:.4f}")
