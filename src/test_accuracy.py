import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from model_definition import MRIClassifier

# Test setinde doğruluk hesaplama fonksiyonu
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Modeli ve test setini yükle
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modeli yükle
    model = MRIClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load("models/BestValAccModel_Epoch50_BatchSz32.pth", map_location=device))

    # Test veri setini yükle
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_data = ImageFolder(root="./data/Raw/Testing", transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    # Modelin doğruluğunu hesapla
    accuracy = test_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")




if __name__ == "__main__":
    main()
