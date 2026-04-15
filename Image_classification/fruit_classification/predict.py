import torch
from torchvision import transforms, models
from PIL import Image
import json

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Define image transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load('plant_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Predict function
def predict_fruit(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()
        class_name = class_names[class_index]

    return class_name

# Example usage
if __name__ == "__main__":
    test_image = "test_image.jpg"  # Replace with your image
    prediction = predict_fruit(test_image)
    print("Predicted Fruit Class:", prediction)
