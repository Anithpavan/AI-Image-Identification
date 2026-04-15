import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import json
from tqdm import tqdm

# ==========================
# DEVICE CONFIGURATION
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================
# PATHS - CHANGE HERE IF NEEDED
# ==========================
dataset_dir = "Birds_dataset"  # root dataset folder on Kaggle
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "valid")
test_dir = os.path.join(dataset_dir, "test")

# ==========================
# DATA TRANSFORMS
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

# ==========================
# DATASETS
# ==========================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

print(f"✔ Training images: {len(train_dataset)}")
print(f"✔ Validation images: {len(val_dataset)}")
print(f"✔ Test images: {len(test_dataset)}")
print(f"✔ Classes: {train_dataset.classes[:10]}... (total {len(train_dataset.classes)})")

# ==========================
# DATALOADERS
# ==========================
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Save class names
class_names = train_dataset.classes
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# ==========================
# MODEL
# ==========================
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze feature extractor

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_names))
)
model = model.to(device)

# ==========================
# LOSS & OPTIMIZER
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ==========================
# TRAINING LOOP
# ==========================
num_epochs = 10
print("🚀 Starting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    train_acc = 100 * correct / len(train_dataset)

    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_acc = 100 * val_correct / len(val_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}%")

# ==========================
# SAVE MODEL
# ==========================
torch.save(model.state_dict(), "bird_model.pth")
print("✅ Model saved as bird_model.pth")

# ==========================
# TEST EVALUATION
# ==========================
model.eval()
test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)
test_acc = 100 * test_correct / len(test_dataset)
print(f"📊 Final Test Accuracy: {test_acc:.2f}%")
