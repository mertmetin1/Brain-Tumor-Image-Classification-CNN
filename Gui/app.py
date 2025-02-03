from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


import torch.nn as nn
class MRIClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MRIClassifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.fc_layers(x)
        return x


# Flask uygulaması
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Model ve sınıflar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MRIClassifier(num_classes=4).to(device)
model.load_state_dict(torch.load("D:\VSCodeWorkSpace\AI\CNNMRIBrainTumorClassificaiton\models\BestValAccModel_Epoch50_BatchSz32.pth", map_location=device))  # Model dosyasını yükleyin
model.eval()
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Dosya yolu ve model tahmini
def predict_and_visualize(image_path):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    # Tahmin yap
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    # Grad-CAM
    target_layers = [model.cnn_layers[-3]]  # Hedef katman
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor)[0]
    img = (tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2  # Görüntüyü normalize et
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return predicted_class, visualization

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Tahmin ve Grad-CAM
        predicted_class, visualization = predict_and_visualize(filepath)
        result_path = os.path.join(app.config['RESULT_FOLDER'], f"result_{predicted_class}_{file.filename}")
        Image.fromarray(visualization).save(result_path)

        return render_template('results.html', predicted_class=predicted_class, image_path=result_path)

if __name__ == '__main__':
    app.run(debug=True)
