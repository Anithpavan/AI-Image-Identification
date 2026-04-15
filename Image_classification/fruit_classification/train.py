import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
dataset_dir = 'C:\\Users\\aluru\\OneDrive\\Desktop\\11plant identification project\\split_ttv_dataset_type_of_plants'

train_dir = os.path.join(dataset_dir, 'Train_Set_Folder')
val_dir = os.path.join(dataset_dir, 'Validation_Set_Folder')
# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
print(f"✔ Loaded {len(train_dataset)} training images from: {train_dir}")
print(f"✔ Loaded {len(val_dataset)} validation images from: {val_dir}")
print(f"✔ Classes: {train_dataset.classes}")

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Save class names for prediction
class_names = train_dataset.classes
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

# Load a pre-trained model and modify final layer
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze base layers

model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Update classifier
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
print("🚀 Starting training...")

# Training loop
num_epochs = 1
from tqdm import tqdm

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


# Save model
torch.save(model.state_dict(), 'plant_model.pth')
print("Training completed and model saved as plant_model.pth")
print("✅ Model saved to plant_model.pth")
