# %%
# Tugas Eksplorasi ResNet
# Team PhilosoPy
# Anggota:
# - Alfajar (122140122)
# - Ikhsannudin Lathief (122140137)
# - Shintya Ayu Wardhani (122140138)

# %% [markdown]
# ## Dependensi

# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from timm.optim.lion import Lion

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchinfo import summary

# %% [markdown]
# ## Dataloader

# %%
class FoodDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['label']]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_dataset():
    csv_path = "train.csv"
    img_dir = "train"
    if not os.path.exists(csv_path):
        print("Dataset not found!")
        return None, None

    df_all = pd.read_csv(csv_path)
    df_sample = df_all.sample(n=min(300, len(df_all)), random_state=Config.RANDOM_STATE)
    train_size = int(0.8 * len(df_sample))
    df_shuffled = df_sample.sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
    train_df = df_shuffled[:train_size]
    val_df = df_shuffled[train_size:]
    train_df.to_csv('temp_train.csv', index=False)
    val_df.to_csv('temp_val.csv', index=False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
    ])

    train_dataset = FoodDataset('temp_train.csv', img_dir, transform)
    val_dataset = FoodDataset('temp_val.csv', img_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    os.remove('temp_train.csv')
    os.remove('temp_val.csv')
    return train_loader, val_loader


# %% [markdown]
# ## Tahap 1 - Plain34

# %%
# =========================
# HYPERPARAMETERS (TAHAP 1)
# =========================
class Config:
    NUM_CLASSES = 5
    LR = 0.0001
    EPOCHS = 20
    BATCH_SIZE = 16
    WEIGHT_DECAY = 1e-4
    RANDOM_STATE = 42
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]


# =========================
# MODEL: PLAIN-34 (NO SKIP)
# =========================
class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.relu(out)
        return out


class Plain34(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(Plain34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [PlainBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_plain34(num_classes=Config.NUM_CLASSES):
    return Plain34(num_classes=num_classes)


# =========================
# TRAINING
# =========================
def setup_training_components(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    return criterion, optimizer


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, device, tag="plain34_baseline"):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    model.to(device)

    for e in range(1, Config.EPOCHS + 1):
        # --- Training dan Validation ---
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        # --- Simpan ke history ---
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # --- Simpan model terbaik ---
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), f"{tag}_best.pth")

        print(
            f"{tag} | Epoch {e}/{Config.EPOCHS} "
            f"| Train Acc: {tr_acc:.2f}% | Train Loss: {tr_loss:.4f} "
            f"| Val Acc: {va_acc:.2f}% | Val Loss: {va_loss:.4f} "
            f"| Best Val Acc: {best_val:.2f}%"
        )

    return history, best_val


# =========================
# SINGLE-FIGURE (TRAIN vs VAL)
# =========================
def plot_tahap1_plain34(history, save_path="tahap1_plain34_train_val.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Tahap 1 — Plain-34 (No Skip): Train vs Val", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], linestyle="--", label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"], linestyle="--", label="Val Acc")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# MAIN (TAHAP 1 SAJA)
# =========================
if __name__ == "__main__":
    print("Tahap 1: Plain-34 Baseline (No Skip Connections)")
    train_loader, val_loader = load_dataset()
    if train_loader is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model = create_plain34()

        # ---- SUMMARY DAN JUMLAH PARAMETER ----
        try:
            summary(model, input_size=(1, 3, 224, 224), verbose=1)
        except Exception as e:
            print(f"torchinfo summary error: {e}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")

        criterion, optimizer = setup_training_components(model)
        history, best_val = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, tag="plain34_baseline"
        )

        plot_tahap1_plain34(history, save_path="tahap1_plain34_train_val.png")

        print("\nRINGKASAN TAHAP 1")
        print(f"Final Val Acc: {history['val_acc'][-1]:.2f}% | Best Val Acc: {best_val:.2f}%")

# %% [markdown]
# ## Tahap 2 - ResNet34

# %%
# =========================
# MODEL: RESNET-34 (WITH SKIP)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        # Implementasi Skip Connection
        out = F.relu(out + identity)
        return out


class ResNet34(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, None))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# =========================
# TRAINING
# =========================
def setup_training_components(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    return criterion, optimizer


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, device, tag="plain34_baseline"):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    model.to(device)

    for e in range(1, Config.EPOCHS + 1):
        # --- Training dan Validation ---
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        # --- Simpan ke history ---
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # --- Simpan model terbaik ---
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), f"{tag}_best.pth")

        print(
            f"{tag} | Epoch {e}/{Config.EPOCHS} "
            f"| Train Acc: {tr_acc:.2f}% | Train Loss: {tr_loss:.4f} "
            f"| Val Acc: {va_acc:.2f}% | Val Loss: {va_loss:.4f} "
            f"| Best Val Acc: {best_val:.2f}%"
        )

    return history, best_val


# =========================
# PLOT: TRAIN vs VAL (1 FIGUR)
# =========================
def plot_resnet34_train_val(history, save_path="tahap2_resnet34_train_val.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ResNet-34 (with Skip) — Train vs Val", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], linestyle="--", label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"], linestyle="--", label="Val Acc")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# MAIN — RESNET34
# =========================
if __name__ == "__main__":
    print("Tahap 2 (ResNet34): Implementasi Residual Connection — ResNet-34")
    train_loader, val_loader = load_dataset()
    if train_loader is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model = ResNet34()

        try:
            summary(model, input_size=(1, 3, 224, 224), verbose=1)
        except Exception as e:
            print(f"torchinfo summary (ResNet34) error: {e}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[ResNet34] Total params: {total_params:,} | Trainable: {trainable_params:,}\n")

        criterion, optimizer = setup_training_components(model)
        history, best_val = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, tag="tahap2_resnet34"
        )

        plot_resnet34_train_val(history, save_path="tahap2_resnet34_train_val.png")

        print("\nRINGKASAN RESNET34 (TAHAP 2)")
        print(f"Final Val Acc: {history['val_acc'][-1]:.2f}% | Best Val Acc: {best_val:.2f}%")


# %% [markdown]
# ## Tahap 3 - Modifikasi ResNet

# %%
# =========================
# HYPERPARAMETERS 
# =========================
class Config:
    NUM_CLASSES = 5
    LR = 0.0001
    EPOCHS = 20
    BATCH_SIZE = 16
    WEIGHT_DECAY = 1e-4
    RANDOM_STATE = 42
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    DROPOUT_P = 0.5

# =========================
# MODEL: RESNET-34 (WITH SKIP)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # Mengganti Fungsi Aktivasi Jadi Leacky ReLU
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=True)   
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = F.leaky_relu(out + identity, negative_slope=0.01, inplace=True)
        return out



class ResNet34Modified(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(ResNet34Modified, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Menambah Dropout
        self.dropout = nn.Dropout(p=Config.DROPOUT_P)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, None))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=True)     
        x = self.maxpool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



# =========================
# TRAINING
# =========================
def setup_training_components(model):
    criterion = nn.CrossEntropyLoss()
    # Mengubah Optimizier jadi Lion
    optimizer = Lion(model.parameters(), lr=3e-4, weight_decay=5e-2, betas=(0.95, 0.98))
    return criterion, optimizer


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, device, tag="plain34_baseline"):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    model.to(device)

    for e in range(1, Config.EPOCHS + 1):
        # --- Training dan Validation ---
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        # --- Simpan ke history ---
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # --- Simpan model terbaik ---
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), f"{tag}_best.pth")

        print(
            f"{tag} | Epoch {e}/{Config.EPOCHS} "
            f"| Train Acc: {tr_acc:.2f}% | Train Loss: {tr_loss:.4f} "
            f"| Val Acc: {va_acc:.2f}% | Val Loss: {va_loss:.4f} "
            f"| Best Val Acc: {best_val:.2f}%"
        )

    return history, best_val


# =========================
# PLOT: TRAIN vs VAL (1 FIGUR)
# =========================
def plot_resnet34_train_val(history, save_path="tahap3_resnet34_train_val.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ResNet-34 (Act: LeakyReLU, Dropout: {Config.DROPOUT_P}, Lion) — Train vs Val",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], linestyle="--", label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train Acc")
    ax.plot(epochs, history["val_acc"], linestyle="--", label="Val Acc")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# MAIN — RESNET34
# =========================
if __name__ == "__main__":
    print(f"Tahap 3 (ResNet34): Residual + Dropout {Config.DROPOUT_P} + Activation: LeakyReLU + Optimizer : Lion")
    train_loader, val_loader = load_dataset()
    if train_loader is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model = ResNet34Modified()

        try:
            summary(model, input_size=(1, 3, 224, 224), verbose=1)
        except Exception as e:
            print(f"torchinfo summary (ResNet34) error: {e}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[ResNet34] Total params: {total_params:,} | Trainable: {trainable_params:,}\n")

        criterion, optimizer = setup_training_components(model)
        history, best_val = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, tag="tahap2_resnet34"
        )

        plot_resnet34_train_val(history, save_path="tahap3_resnet34_modified_train_val.png")

        print("\nRINGKASAN RESNET34_MODIFIED (TAHAP 3)")
        print(f"Final Val Acc: {history['val_acc'][-1]:.2f}% | Best Val Acc: {best_val:.2f}%")
