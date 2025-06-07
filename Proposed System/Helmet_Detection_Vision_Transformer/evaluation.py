import os
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import label_binarize

# Paths and configurations
model_dir = "./models/helmet_vit/"
train_csv = "./train_labels.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and preprocessor
model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)
processor = ViTImageProcessor.from_pretrained(model_dir)
model.to(device)
model.eval()

# Load dataset
df = pd.read_csv(train_csv)
print("Available columns:", df.columns)

# Extract image paths and labels
image_paths = df['image_path'].tolist()
labels = df['label'].tolist()
label_classes = sorted(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(label_classes)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

# Define dataset class
class HelmetDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label_to_idx[label]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset and DataLoader
dataset = HelmetDataset(image_paths, labels, transform)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, targets in dataloader:
        images = images.to(device)
        targets = torch.tensor(targets).to(device)
        outputs = model(images).logits
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

# Convert predictions and labels back to class names
all_preds = pd.Series(all_preds).map(idx_to_label).values
all_labels = pd.Series(all_labels).map(idx_to_label).values

# Classification report
print(classification_report(all_labels, all_preds))

# Visualizations
# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=label_classes)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(label_classes)), label_classes, rotation=45)
plt.yticks(range(len(label_classes)), label_classes)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# ROC Curve
if len(label_classes) > 2:
    y_true_bin = label_binarize([label_to_idx[label] for label in all_labels], classes=range(len(label_classes)))
    y_pred_bin = label_binarize([label_to_idx[label] for label in all_preds], classes=range(len(label_classes)))
    for i, class_name in enumerate(label_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")
else:
    y_true_bin = label_binarize([label_to_idx[label] for label in all_labels], classes=range(len(label_classes)))
    y_pred_bin = [label_to_idx[label] for label in all_preds]
    fpr, tpr, _ = roc_curve(y_true_bin[:, 1], y_pred_bin)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true_bin[:, 1], y_pred_bin)
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()



'''
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Load model and preprocessor
model_path = './models/helmet_vit/'
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path, config=config)
processor = AutoImageProcessor.from_pretrained(model_path)

# Custom Dataset class
class HelmetDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Open image
        image = Image.open(img_path).convert("RGB")

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['label'] = torch.tensor(label, dtype=torch.long)
        return inputs

# Load test dataset
test_csv = "./test_labels.csv"  # Path to your test.csv file
test_dataset = HelmetDataset(test_csv, processor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate model
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: value.squeeze() for key, value in batch.items() if key != 'label'}
            labels = batch['label']
            outputs = model(**inputs)

            preds = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    return np.array(all_labels), np.array(all_preds)

# Get predictions
labels, preds = evaluate_model(model, test_loader)

# Classification report
print("Classification Report:\n")
print(classification_report(labels, preds))

# Confusion matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.id2label.values())
disp.plot(cmap='viridis')
plt.title("Confusion Matrix")
plt.show()
'''
