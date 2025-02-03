from datetime import datetime
import time

import os
import torch
from torch import nn, optim
from visualization import  save_class_distribution_piechart,visualize_sample_images
from model_definition import MRIClassifier
from data_preparation import prepare_data
from train_and_evaluate import train_model, evaluate_model

# Ayarlar
data_path = "./data/Raw"
batch_size = 32
num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri hazırlığı
train_loader, val_loader, test_loader, classes = prepare_data(data_path, batch_size)

# Eğitim seti
save_class_distribution_piechart(
    train_loader.dataset, classes, 
    "Training Set Class Distribution", 
    "training_set_distribution"
)

# Doğrulama seti
save_class_distribution_piechart(
    val_loader.dataset, classes, 
    "Validation Set Class Distribution", 
    "validation_set_distribution"
)

# Test seti
save_class_distribution_piechart(
    test_loader.dataset, classes, 
    "Test Set Class Distribution", 
    "test_set_distribution"
    
)


visualize_sample_images(train_loader.dataset, classes, title="Training Set Sample Images")


# Model
model = MRIClassifier(num_classes=len(classes)).to(device)



from torchviz import make_dot
# Visualize Model Architecture
dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Example input
model_graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
os.makedirs('./plots', exist_ok=True)
model_graph.render("./plots/model_architecture", format="png", cleanup=True)
print("Model architecture diagram saved to './plots/model_architecture.png'")


# Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
val_accuracies = []

# Start time for training
start_time = time.time()
# Eğitim
best_accuracy = 0.0
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # En iyi modeli kaydet
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        os.makedirs('./models', exist_ok=True) 
        filename = f"./models/BestValAccModel_Epoch{num_epochs}_BatchSz{batch_size}.pth"
        torch.save(model.state_dict(), filename)
        print(f"New best model saved at: {filename} accuracy: {best_accuracy:.2f}%")

        
end_time = time.time()
training_time = end_time - start_time

import matplotlib.pyplot as plt

# Kayıpları çizdir
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
# Save figure
os.makedirs('./plots', exist_ok=True)  # '/plots' dizini oluştur
plt.savefig(f'./plots/Training vs Validation Loss.png')
plt.close()





# Doğruluk çizdir
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Epochs")
plt.legend()
# Save figure
os.makedirs('./plots', exist_ok=True)  # '/plots' dizini oluştur
plt.savefig(f'./plots/Validation Accuracy Over Epochs.png')
plt.close()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Test setinde tahminler
y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig('./plots/confusion_matrix.png')
plt.close()


import numpy as np

cm = confusion_matrix(y_true, y_pred)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
plt.bar(classes, class_accuracies)
plt.xlabel("Classes")
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.savefig('./plots/per_class_accuracy.png')
plt.close()



from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Sınıfları binarize et
y_true_bin = label_binarize(y_true, classes=range(len(classes)))
y_pred_proba = []

# Tahmin edilen olasılıkları topla
model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_pred_proba.extend(outputs.cpu().numpy())

y_pred_proba = np.array(y_pred_proba)

# ROC eğrisi
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC eğrilerini çizdir
colors = cycle(["blue", "red", "green", "orange", "purple"])
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig('./plots/roc_curve.png')
plt.close()


import random
# Doğru ve yanlış tahminler için verileri alın
correct_preds = [
    (test_loader.dataset[i][0], test_loader.dataset[i][1]) 
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) 
    if true_label == pred_label
]

incorrect_preds = [
    (test_loader.dataset[i][0], test_loader.dataset[i][1], pred_label) 
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)) 
    if true_label != pred_label
]

os.makedirs('./plots/Predictions', exist_ok=True)  # '/plots' dizini oluştur
# Doğru tahminlerden örnekler
for i in range(5):  # 5 örnek
    img, label = random.choice(correct_preds)
    img = img.permute(1, 2, 0).numpy()  # Tensor'u görselleştirme için dönüştürün
    plt.imshow(img)
    plt.title(f"True Label: {classes[label]}")
    plt.axis("off")
    plt.savefig(f'./plots/Predictions/correct_prediction_{i}.png')
    plt.close()

# Yanlış tahminlerden örnekler
for i in range(5):  # 5 örnek
    img, true_label, pred_label = random.choice(incorrect_preds)
    img = img.permute(1, 2, 0).numpy()  # Tensor'u görselleştirme için dönüştürün
    plt.imshow(img)
    plt.title(f"True: {classes[true_label]}, Pred: {classes[pred_label]}")
    plt.axis("off")
    plt.savefig(f'./plots/Predictions/incorrect_prediction_{i}.png')
    plt.close()



from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)

# Precision ve Recall görselleştirme
x = np.arange(len(classes))
width = 0.35

plt.bar(x - width/2, precision, width, label='Precision')
plt.bar(x + width/2, recall, width, label='Recall')
plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Precision and Recall per Class')
plt.xticks(x, classes)
plt.legend()
plt.savefig('./plots/precision_recall_per_class.png')
plt.close()

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
disp.plot(cmap="Blues")
plt.title("Normalized Confusion Matrix")
plt.savefig('./plots/normalized_confusion_matrix.png')
plt.close()

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# Grad-CAM için hedef katman
target_layers = [model.cnn_layers[-3]]

# Grad-CAM nesnesini oluştur
cam = GradCAM(model=model, target_layers=target_layers)

# 5 farklı örnek üzerinde Grad-CAM görselleştirmesi
for idx in range(5):  # İlk 5 örneği işlemek için döngü
    inputs, labels = next(iter(test_loader))  # Test veri kümesinden veri alın
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Grad-CAM hesapla
    grayscale_cam = cam(input_tensor=inputs)[idx, :]  # idx ile örneği seç
    
    # Görselleştirme için Tensor'u normalize et
    img = inputs[idx].cpu().numpy().transpose(1, 2, 0)  # Tensor -> NumPy
    img = (img - img.min()) / (img.max() - img.min())  # Normalize
    
    # Grad-CAM görselleştirmesi
    visualization = show_cam_on_image(img, grayscale_cam)
    
    # Görüntüyü kaydet
    plt.imshow(visualization)
    plt.axis('off')
    plt.title(f"Grad-CAM Visualization Example {idx+1}")
    os.makedirs('./plots/grad_cam', exist_ok=True)  # '/plots' dizini oluştur
    plt.savefig(f'./plots/grad_cam/grad_cam_example_{idx+1}.png')
    plt.close()

print("Grad-CAM görselleştirmeleri './plots/' klasörüne kaydedildi.")





probs = torch.softmax(torch.tensor(y_pred_proba), dim=1)
uncertainties = 1 - probs.max(dim=1).values
plt.hist(uncertainties.numpy(), bins=20)
plt.xlabel('Uncertainty')
plt.ylabel('Frequency')
plt.title('Prediction Uncertainty Distribution')
plt.savefig('./plots/prediction_uncertainty.png')
plt.close()

os.makedirs('./plots/weight_distributions', exist_ok=True)  # '/plots' dizini oluştur
for name, param in model.named_parameters():
    if param.requires_grad:
        plt.hist(param.data.cpu().numpy().flatten(), bins=50)
        plt.title(f'Weight Distribution: {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(f'./plots/weight_distributions/weight_distribution_{name}.png')
        plt.close()

import pandas as pd

# Model ile ilgili bilgileri derleyin
model_metrics = {
    "Hyperparameters": [
        "Batch Size",
        "Number of Epochs",
        "Learning Rate",
        "Number of Classes",
        "Device"
    ],
    "Values": [
        batch_size,
        num_epochs,
        learning_rate,
        len(classes),
        str(device)
    ]
}


model_summary = str(model)
performance_metrics = {
    "Metrics": [
        "Best Validation Accuracy",
        "Final Validation Accuracy",
        "Final Training Loss",
        "Final Validation Loss",
        "ROC-AUC (per class)",
        "Training Time (s)", 
        "Model Details"
    ],
    "Values": [
        f"{best_accuracy:.2f}%",
        f"{val_accuracies[-1]:.2f}%",
        f"{train_losses[-1]:.4f}",
        f"{val_losses[-1]:.4f}",
        {classes[i]: f"{roc_auc[i]:.2f}" for i in range(len(classes))},
        f"{training_time:.2f}",
        model_summary
    ]
}

# Verileri birleştirip tablo oluşturun
data = {
    "Category": model_metrics["Hyperparameters"] + performance_metrics["Metrics"],
    "Value": model_metrics["Values"] + performance_metrics["Values"]
}

df = pd.DataFrame(data)

# Tarih ve saat etiketi
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = f"./model_metrics_{current_time}.csv"

# CSV olarak kaydet
os.makedirs('./metrics', exist_ok=True)  # Klasör oluştur
output_path = os.path.join('./metrics', f"model_metrics_{current_time}.csv")
df.to_csv(output_path, index=False)
print(f"Model bilgileri ve metrikler {output_path} dosyasına kaydedildi.")

