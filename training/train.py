from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import DeepfakeCNN

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "dataset" / "splits"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "deepfake_model.pth"
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42
LABELS = {"real": 0, "fake": 1}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class FrameDataset(Dataset):
    def __init__(self, split, augment=False):
        self.items = []
        self.augment = augment
        for label, target in LABELS.items():
            folder = SPLITS / split / label
            if folder.exists():
                self.items.extend((p, target) for p in sorted(folder.glob("*.jpg")))
        if not self.items:
            raise RuntimeError(f"No images found for split '{split}'. Run prepare_dataset.py first.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.35:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor, torch.tensor(label, dtype=torch.long)

def main():
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")
    train_ds = FrameDataset("train", augment=True)
    val_ds = FrameDataset("val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepfakeCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    best_val_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Training frames: {len(train_ds)} | Validation frames: {len(val_ds)}")
    print(f"Starting {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": LABELS,
                "image_size": IMAGE_SIZE,
                "architecture": "DeepfakeCNN",
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            }, MODEL_PATH)
            print(f"  Saved best checkpoint: {MODEL_PATH}")

    print("Training finished.")
    print("This is a small-dataset prototype; validation accuracy is not proof of real-world detection.")

if __name__ == "__main__":
    main()
