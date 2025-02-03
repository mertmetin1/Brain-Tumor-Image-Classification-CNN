import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from model_definition import MRIClassifier

def predict_image(image_path, model, classes, device):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# Modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRIClassifier(num_classes=4).to(device)
model.load_state_dict(torch.load("D:\VSCodeWorkSpace\AI\CNNMRIBrainTumorClassificaiton\models\\BestValAccModel_Epoch50_BatchSz32.pth"))
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Test
image_path = "D:\VSCodeWorkSpace\AI\CNNMRIBrainTumorClassificaiton\data\Raw\Testing\glioma\Te-gl_0010.jpg"  # Örnek bir test görüntüsü
predicted_class = predict_image(image_path, model, classes, device)
print(f"Predicted Class: {predicted_class}")
