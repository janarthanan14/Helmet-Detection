import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Step 1: Dataset Preparation
class HelmetDataset(Dataset):
    def __init__(self, csv_file, processor, base_dir=None):
        """
        Initialize the dataset from a CSV file.
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            processor: The image processor for preprocessing.
            base_dir (str): Optional base directory to prepend to relative paths.
        """
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.base_dir = base_dir

        # Validate and correct paths
        self.data['image_path'] = self.data['image_path'].apply(self.validate_path)
        if self.data['image_path'].isnull().any():
            raise ValueError("No valid image paths found in the CSV.")

    def validate_path(self, image_path):
        """
        Validate and correct image paths.
        Args:
            image_path (str): The original image path from the CSV.
        Returns:
            str: The valid, corrected image path.
        """
        if self.base_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.base_dir, image_path)

        if not os.path.isfile(image_path):
            print(f"Invalid path: {image_path}")
            return None
        return image_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data point.
        Args:
            idx (int): Index of the data point.
        Returns:
            dict: Processed image tensor and label.
        """
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
        return inputs['pixel_values'], label

# Step 2: Configuration
# Paths to CSV files
train_csv_path = "./train_labels.csv"  # CSV with train image paths and labels
test_csv_path = "./test_labels.csv"    # CSV with test image paths and labels
base_dir = "D:/demo/" #/dataset/train/images"  # Base directory for image paths

# Initialize the image processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Create datasets and DataLoaders
train_dataset = HelmetDataset(train_csv_path, processor, base_dir=base_dir)
test_dataset = HelmetDataset(test_csv_path, processor, base_dir=base_dir)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 3: Model Initialization
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,  # Binary classification
    ignore_mismatched_sizes=True
)
model = model.to(device)

# Step 4: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Step 5: Training Loop
epochs = 5  # Adjust based on requirements
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for pixel_values, labels in train_loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Step 6: Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for pixel_values, labels in test_loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixel_values)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Step 7: Save the Fine-Tuned Model
output_dir = "./helmet_detection_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("Model and processor saved successfully.")

# Step 8: Inference
def predict(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'])
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_class

# Test inference
test_image = "./dataset/test/images/BikesHelmets102_png_jpg.rf.6bc6d292ee31801b348ecd1b3ff56e58.jpg"  # Replace with your test image path
predicted_class = predict(test_image)
print(f"Predicted class for the test image: {predicted_class}")  # 0: no helmet, 1: helmet



'''
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Step 1: Dataset Preparation
class HelmetDataset(Dataset):
    def __init__(self, csv_file, processor):
        """
        Initialize the dataset from a CSV file.
        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            processor: The image processor for preprocessing.
        """
        self.data = pd.read_csv(csv_file)
        self.processor = processor

        # Validate paths and filter out missing files
        self.data = self.data[self.data['image_path'].apply(os.path.isfile)].reset_index(drop=True)
        if self.data.empty:
            raise ValueError("No valid image paths found in the CSV.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data point.
        Args:
            idx (int): Index of the data point.
        Returns:
            dict: Processed image tensor and label.
        """
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
        return inputs['pixel_values'], label

# Step 2: Configuration
# Paths to CSV files
train_csv_path = "./train_labels.csv"  # CSV with train image paths and labels
test_csv_path = "./test_labels.csv"    # CSV with test image paths and labels

# Example structure of the CSV:
# image_path,label
# /path/to/image1.jpg,1
# /path/to/image2.jpg,0

# Initialize the image processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Create datasets and DataLoaders
train_dataset = HelmetDataset(train_csv_path, processor)
test_dataset = HelmetDataset(test_csv_path, processor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 3: Model Initialization
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,  # Binary classification
    ignore_mismatched_sizes=True
)
model = model.to(device)

# Step 4: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Step 5: Training Loop
epochs = 5  # Adjust based on requirements
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for pixel_values, labels in train_loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Step 6: Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for pixel_values, labels in test_loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixel_values)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Step 7: Save the Fine-Tuned Model
output_dir = "./helmet_detection_model"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("Model and processor saved successfully.")

# Step 8: Inference
def predict(image_path):
    model.eval()
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'])
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_class

# Test inference
test_image = "./datasets/test/images/BikesHelmets102_png_jpg.rf.6bc6d292ee31801b348ecd1b3ff56e58.jpg"  # Replace with your test image path
predicted_class = predict(test_image)
print(f"Predicted class for the test image: {predicted_class}")  # 0: no helmet, 1: helmet
'''
