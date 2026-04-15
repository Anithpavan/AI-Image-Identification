import torch
from torchvision import transforms, models
from PIL import Image
import json
import os

# ==========================
# LOAD CLASS NAMES
# ==========================
if not os.path.exists("class_names.json"):
    raise FileNotFoundError("❌ class_names.json not found! Please run train.py first.")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ==========================
# IMAGE TRANSFORM (same as validation)
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================
# LOAD TRAINED MODEL
# ==========================
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(256, len(class_names))
    )
    model.load_state_dict(torch.load("bird_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_animal(image_path):
    """
    Takes an image path, predicts bird species, 
    and returns (species_name, details_dict)
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]

    # Optional: Load additional info if available
    if os.path.exists("bird_info.json"):
        with open("bird_info.json", "r", encoding="utf-8") as f:
            bird_info = json.load(f)
        details = bird_info.get(class_name, {
            "Scientific Name": "Unknown",
            "Habitat": "Unknown",
            "Diet": "Unknown",
            "Description": "No details available"
        })
    else:
        details = {
            "Scientific Name": "Unknown",
            "Habitat": "Unknown",
            "Diet": "Unknown",
            "Description": "No extra info available"
        }

    return class_name, details

# ==========================
# DEBUG / TEST MODE
# ==========================
if __name__ == "__main__":
    test_img = "test.jpg"  # replace with any bird image path
    bird, info = predict_animal(test_img)
    print("Prediction:", bird)
    print("Details:", info)
