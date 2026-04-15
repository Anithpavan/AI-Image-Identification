import torch
from torchvision import transforms, models
from PIL import Image
import json

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load breed info
with open("breed_info.json", "r", encoding="utf-8") as f:
    breed_info = json.load(f)

# Transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load("animal_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Prediction function
def predict_animal(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted.item()]

    # Get details from breed_info.json
    details = breed_info.get(class_name, {
        "Origin": "Unknown",
        "Climatic Conditions": "Unknown",
        "Use": "Unknown"
    })

    return class_name, details

# Debug/test
if __name__ == "__main__":
    test_img = "test.jpg"  # replace with your test image path
    breed, info = predict_animal(test_img)
    print("Prediction:", breed)
    print("Details:", info)
