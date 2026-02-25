import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import build_model

# ── Cihaz ayarı (Apple Silicon için MPS) ──────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# ── Transform'lar ─────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Veri setleri ──────────────────────────────────────────────────
train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("data/valid", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

print(f"Sınıf sayısı: {len(train_dataset.classes)}")
print(f"Eğitim görüntü sayısı: {len(train_dataset)}")
print(f"Validasyon görüntü sayısı: {len(val_dataset)}")

# Sınıf isimlerini kaydet (app.py'de kullanacağız)
import json
with open("models/class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)

# ── Model ─────────────────────────────────────────────────────────
num_classes = len(train_dataset.classes)
model = build_model(num_classes).to(device)

# ── Loss ve Optimizer ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Eğitim ve Validasyon Fonksiyonları ────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in tqdm(loader, desc="Eğitim"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def val_epoch(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validasyon"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


# ── Eğitim Döngüsü ────────────────────────────────────────────────
NUM_EPOCHS = 10
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc     = val_epoch(model, val_loader)
    scheduler.step()

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("  ✅ Model kaydedildi!")

print(f"\nEğitim tamamlandı! En iyi Val Acc: {best_val_acc:.3f}")