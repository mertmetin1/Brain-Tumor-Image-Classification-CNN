# MRI TUMOR CLASSIFICATION

## 📌 Overview
This project implements a **deep learning-based MRI tumor classification model** using **PyTorch**. The model is trained on MRI brain scans to classify different tumor types using a **CNN architecture**.

## 🚀 Features
- **Preprocessing**: Image transformations, data augmentation, and dataset splitting.
- **Deep Learning Model**: Custom Convolutional Neural Network (**CNN**) for MRI classification.
- **Training & Evaluation**: Model training, validation, and evaluation metrics.
- **Visualization**: Data distribution, confusion matrix, class performance, and Grad-CAM heatmaps.
- **Testing & Prediction**: Evaluate model accuracy and classify new MRI images.

---
## 📂 Project Structure
```
├── data_preparation.py   # Prepares dataset, applies transformations
├── model_definition.py   # Defines CNN model
├── train_and_evaluate.py # Training & evaluation logic
├── test_accuracy.py      # Calculates test set accuracy
├── testSample.py         # Predicts MRI tumor class for a given image
├── main.py               # Orchestrates data processing, training & testing
├── visualization.py      # Contains visualization functions
├── models/               # Stores trained model weights
├── plots/                # Stores various plots (confusion matrix, ROC curves, etc.)
└── metrics/              # Stores model performance metrics
```

---
## 🔧 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/mertmetin1/MRI-Tumor-Classification.git
cd MRI-Tumor-Classification
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Dataset Preparation**
Ensure the dataset is structured as follows:
```
./data/Raw/
 ├── Training/
 │   ├── glioma/
 │   ├── meningioma/
 │   ├── notumor/
 │   ├── pituitary/
 ├── Testing/
 │   ├── glioma/
 │   ├── meningioma/
 │   ├── notumor/
 │   ├── pituitary/
```

---
## 🏋️‍♂️ Training the Model
Run the `main.py` script to train the model:
```bash
python main.py
```
This will:
- Load the dataset
- Train the CNN model
- Save the best-performing model
- Generate evaluation plots

---
## 🧪 Testing & Prediction
### **Test Model Accuracy**
```bash
python test_accuracy.py
```

### **Classify a New MRI Image**
```bash
python testSample.py --image_path path/to/image.jpg
```

---
## 📊 Visualization & Evaluation
- **Confusion Matrix** (`plots/confusion_matrix.png`)
- ![resim](https://github.com/user-attachments/assets/266b0b52-3af6-4cbb-8f33-508e0fd7b666)

- **ROC Curves** (`plots/roc_curve.png`)
- ![resim](https://github.com/user-attachments/assets/668599fc-3603-45f1-9fb4-d6896d871ded)

- **Class-wise Performance** (`plots/per_class_accuracy.png`)
- ![resim](https://github.com/user-attachments/assets/590782b5-b116-4e71-8b16-de2aef67ae4d)

- **Grad-CAM Heatmaps** (`plots/grad_cam/`)
- ![resim](https://github.com/user-attachments/assets/29b68acf-0604-4bdb-bdb4-c48688da97a8)



---
## 📈 Model Performance
| Metric                 | Value |
|------------------------|-------|
| Best Validation Accuracy | **97.46%** |
| Final Training Loss   | **0.0272** |
| Final Validation Loss | **0.1888** |
| Test Accuracy        | **98.63%** |

---
## 🤖 Model Architecture
The CNN model consists of **3 convolutional layers**, **ReLU activations**, **MaxPooling**, and **fully connected layers**:
```
Conv2D(3 → 32) → ReLU → MaxPool
Conv2D(32 → 64) → ReLU → MaxPool
Conv2D(64 → 128) → ReLU → MaxPool
Flatten → Fully Connected → Output (Softmax)
```
For a detailed visualization, see `plots/model_architecture.png`.

---
## 📝 License
This project is licensed under the MIT License.

---
## 🙌 Acknowledgments
Special thanks to the open-source community for providing tools like PyTorch & TorchVision for deep learning research.

---
## 🔗 References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)

