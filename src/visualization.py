import matplotlib.pyplot as plt
from collections import Counter
import os
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_curve, auc
from itertools import cycle
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import random

# Grafik kaydetme fonksiyonu
def save_class_distribution_piechart(dataset, class_names, title, filename):
    # Her sınıfın örnek sayısını hesapla
    class_counts = Counter([label for _, label in dataset])
    class_values = [class_counts[i] for i in range(len(class_names))]
    
    # Pie chart çiz ve kaydet
    plt.figure(figsize=(6, 6))
    plt.pie(class_values, labels=class_names, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # Circle
    # Save figure
    os.makedirs('./plots', exist_ok=True)  # '/plots' dizini oluştur
    plt.savefig(f'./plots/{filename}.png')
    plt.close()
    
import matplotlib.pyplot as plt
import numpy as np

def denormalize_image(img):
    img = img * 0.5 + 0.5  # Assuming mean=0.5, std=0.5 for normalization
    return img

def visualize_sample_images(dataset, classes, title="Sample Images", num_images=5):
    fig, axes = plt.subplots(len(classes), num_images, figsize=(15, len(classes) * 3))
    for i, class_name in enumerate(classes):
        class_indices = [idx for idx, (_, label) in enumerate(dataset) if label == i]
        sampled_indices = np.random.choice(class_indices, num_images, replace=False)
        for j, idx in enumerate(sampled_indices):
            img, label = dataset[idx]
            img = denormalize_image(img)  # Normalize edilmiş görüntüyü geri çevir
            axes[i, j].imshow(img.permute(1, 2, 0))  # Tensoru görsele dönüştür
            axes[i, j].set_title(f"{class_name}")
            axes[i, j].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    # Save figure
    os.makedirs('./plots', exist_ok=True)  # '/plots' dizini oluştur
    plt.savefig(f'./plots/{title}.png')
    plt.close()



def plot_training_validation_metrics(train_losses, val_losses, val_accuracies):
    os.makedirs('./plots', exist_ok=True)

    # Plot Training vs Validation Loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig('./plots/training_vs_validation_loss.png')
    plt.close()

    # Plot Validation Accuracy
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.savefig('./plots/validation_accuracy_over_epochs.png')
    plt.close()

def plot_confusion_matrices(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig('./plots/confusion_matrix.png')
    plt.close()

    # Normalized Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.savefig('./plots/normalized_confusion_matrix.png')
    plt.close()

def plot_precision_recall(y_true, y_pred, classes):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    x = np.arange(len(classes))
    width = 0.35

    plt.bar(x - width/2, precision, width, label='Precision')
    plt.bar(x + width/2, recall, width, label='Recall')
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Precision and Recall per Class')
    plt.xticks(x, classes)
    plt.legend()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/precision_recall_per_class.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, classes):
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(["blue", "red", "green", "orange", "purple"])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/roc_curve.png')
    plt.close()

def visualize_predictions(test_loader, y_true, y_pred, classes):
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

    os.makedirs('./plots/predictions', exist_ok=True)
    for i in range(5):  # Correct predictions
        img, label = random.choice(correct_preds)
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"True Label: {classes[label]}")
        plt.axis("off")
        plt.savefig(f'./plots/predictions/correct_prediction_{i}.png')
        plt.close()

    for i in range(5):  # Incorrect predictions
        img, true_label, pred_label = random.choice(incorrect_preds)
        img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"True: {classes[true_label]}, Pred: {classes[pred_label]}")
        plt.axis("off")
        plt.savefig(f'./plots/predictions/incorrect_prediction_{i}.png')
        plt.close()

def visualize_grad_cam(model, test_loader, device):
    target_layers = [model.cnn_layers[-3]]
    cam = GradCAM(model=model, target_layers=target_layers)

    os.makedirs('./plots/grad_cam', exist_ok=True)
    for idx in range(5):
        inputs, labels = next(iter(test_loader))
        inputs, labels = inputs.to(device), labels.to(device)

        grayscale_cam = cam(input_tensor=inputs)[idx, :]
        img = inputs[idx].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())

        visualization = show_cam_on_image(img, grayscale_cam)
        plt.imshow(visualization)
        plt.axis('off')
        plt.title(f"Grad-CAM Visualization Example {idx+1}")
        plt.savefig(f'./plots/grad_cam/grad_cam_example_{idx+1}.png')
        plt.close()



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_cnn_workflow():
    fig, ax = plt.subplots(figsize=(12, 6))

    # Feature Extraction Section
    ax.text(0.1, 0.8, "Input", fontsize=12, ha="center", va="center")
    ax.add_patch(Rectangle((0.05, 0.75), 0.1, 0.1, edgecolor="black", facecolor="skyblue"))
    
    ax.arrow(0.15, 0.8, 0.1, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")
    
    ax.text(0.3, 0.8, "Convolution", fontsize=12, ha="center", va="center")
    ax.add_patch(Rectangle((0.25, 0.75), 0.1, 0.1, edgecolor="black", facecolor="lightgreen"))
    
    ax.arrow(0.35, 0.8, 0.1, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")
    
    ax.text(0.5, 0.8, "Pooling", fontsize=12, ha="center", va="center")
    ax.add_patch(Rectangle((0.45, 0.75), 0.1, 0.1, edgecolor="black", facecolor="orange"))
    
    ax.arrow(0.55, 0.8, 0.1, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")
    
    # Classification Section
    ax.text(0.7, 0.8, "Fully Connected", fontsize=12, ha="center", va="center")
    ax.add_patch(Rectangle((0.65, 0.75), 0.1, 0.1, edgecolor="black", facecolor="pink"))
    
    ax.arrow(0.75, 0.8, 0.1, 0, head_width=0.03, head_length=0.03, fc="black", ec="black")
    
    ax.text(0.9, 0.8, "Output", fontsize=12, ha="center", va="center")
    ax.add_patch(Rectangle((0.85, 0.75), 0.1, 0.1, edgecolor="black", facecolor="red"))
    
    # Labels
    ax.text(0.3, 0.6, "Feature Extraction", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.plot([0.1, 0.6], [0.65, 0.65], "k--")
    
    ax.text(0.8, 0.6, "Classification", fontsize=14, ha="center", va="center", fontweight="bold")
    ax.plot([0.7, 0.9], [0.65, 0.65], "k--")
    
    # Adjust plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1)
    ax.axis("off")
    plt.tight_layout()
    
    # Save diagram
    os.makedirs('./plots', exist_ok=True)
    plt.savefig("./plots/cnn_workflow_diagram.png")
    plt.show()


