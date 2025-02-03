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
git clone https://github.com/YOUR_GITHUB_USERNAME/MRI-Tumor-Classification.git
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
- **ROC Curves** (`plots/roc_curve.png`)
- **Class-wise Performance** (`plots/per_class_accuracy.png`)
- **Grad-CAM Heatmaps** (`plots/grad_cam/`)

---
## 📈 Model Performance
| Metric                 | Value |
|------------------------|-------|
| Best Validation Accuracy | **X.XX%** |
| Final Training Loss   | **X.XX** |
| Final Validation Loss | **X.XX** |
| Test Accuracy        | **X.XX%** |

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

