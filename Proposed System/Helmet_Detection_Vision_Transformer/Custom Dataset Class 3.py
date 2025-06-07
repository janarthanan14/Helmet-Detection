import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars

# 1. Define the ViT Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2  # Binary classification: helmet or no-helmet
)

# 2. Define the Image Processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 3. Detect device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Print model configuration summary
print(f"Model Configuration: {model.config}")
print(f"Using device: {device}")

# 5. Define the Dataset Class
class HelmetDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Get the corresponding label
        label = self.labels[idx]
        
        # Process the image using the ViTImageProcessor
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Return the processed pixel values and the label
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)


# 6. Load CSV data (replace with your actual file paths)
train_df = pd.read_csv('./train_labels.csv')  # Replace with actual CSV file path
val_df = pd.read_csv('./valid_labels.csv')  # Replace with actual CSV file path

# 7. Extract image paths and labels from the DataFrame
train_image_paths = train_df['image_path'].tolist()  # List of paths to train images
train_labels = train_df['label'].tolist()  # List of labels for train images

val_image_paths = val_df['image_path'].tolist()  # List of paths to validation images
val_labels = val_df['label'].tolist()  # List of labels for validation images

# 8. Create Dataset instances
train_data = HelmetDataset(train_image_paths, train_labels, processor)
val_data = HelmetDataset(val_image_paths, val_labels, processor)

# 9. Create DataLoader instances
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# 10. Training loop with tqdm for progress bar
def train(model, train_loader, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0

    # Use tqdm for a progress bar in the training loop
    for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Epoch", ncols=100):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images).logits  # Forward pass
        loss = torch.nn.functional.cross_entropy(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss
'''
# 11. Validation loop with tqdm for progress bar
def validate(model, val_loader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        # Use tqdm for a progress bar in the validation loop
        for batch_idx, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating Epoch", ncols=100):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits  # Forward pass
            loss = torch.nn.functional.cross_entropy(outputs, labels)  # Compute loss

            total_loss += loss.item()

            # Calculate accuracy
            preds = outputs.argmax(dim=1)
            correct_preds += (preds == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_preds / len(val_loader.dataset)
    return avg_loss, accuracy

# 12. Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 13. Example of running training and validation loop
num_epochs = 5  # Set number of epochs
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    # Train the model
    avg_train_loss = train(model, train_loader, optimizer, device)
    print(f"Training Loss: {avg_train_loss:.4f}")
    
    # Validate the model
    avg_val_loss, val_accuracy = validate(model, val_loader, device)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

'''


import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, get_scheduler
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars

# Path to your pre-trained model
model_dir = "./helmet_detection_model/"  # Change to your model directory path

# 1. Load the pre-trained model
model = AutoModelForImageClassification.from_pretrained(model_dir, trust_remote_code=True)

# 2. Load the feature extractor (processor)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)

# 3. Set up device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Set up optimizer and loss function
optimizer = Adam(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# Assuming num_epochs and train_loader are defined already
num_epochs = 5  # Set the number of epochs for training
num_training_steps = len(train_loader) * num_epochs

# 5. Learning rate scheduler
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
'''
# 6. Training loop with tqdm for progress bar
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Wrap train_loader with tqdm to show the progress bar
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", total=len(train_loader), ncols=100)):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)  # Move to GPU if available

        # Forward pass through the model
        outputs = model(images)
        logits = outputs.logits
        loss = loss_fn(logits, labels)  # Calculate the loss
        total_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the learning rate scheduler
        lr_scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Optionally, add validation here if needed
    # ...

'''



from sklearn.metrics import classification_report
from tqdm import tqdm  # Import tqdm for progress bars

# Set the model to evaluation mode
model.eval()

# Initialize lists to store predictions and labels
all_preds, all_labels = [], []

# Disable gradient computation for validation
with torch.no_grad():
    # Wrap val_loader with tqdm for progress bar
    for batch in tqdm(val_loader, desc="Validating", total=len(val_loader), ncols=100):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = model(images)
        preds = torch.argmax(outputs.logits, dim=-1)

        # Store the predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print the classification report
print(classification_report(all_labels, all_preds))




from transformers import pipeline
from tqdm import tqdm
from PIL import Image
import os

# Save the model and feature extractor
model.save_pretrained("./models/helmet_vit")
feature_extractor.save_pretrained("./models/helmet_vit")

# Loading the model and feature extractor for inference
helmet_detector = pipeline("image-classification", model="./models/helmet_vit")

# If you are running inference over multiple images, here's how to show progress
image_paths = ["./test_image1.jpg", "./test_image2.jpg", "./test_image3.jpg"]  # List your test images

# Show progress for inference
results = []
for img_path in tqdm(image_paths, desc="Processing Images", ncols=100):
    result = helmet_detector(img_path)
    results.append((img_path, result))

# Print the inference results
for img_path, result in results:
    print(f"Results for {img_path}: {result}")




'''
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

# Define the ViT Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2  # Binary classification: helmet or no-helmet
)

# Define the Image Processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Detect device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print model configuration summary
print(f"Model Configuration: {model.config}")
print(f"Using device: {device}")

class HelmetDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)

# Load data
train_data = HelmetDataset(train_image_paths, train_labels, feature_extractor)
val_data = HelmetDataset(val_image_paths, val_labels, feature_extractor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
'''

'''
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import os
import pandas as pd

# Define HelmetDataset
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

# Define Vision Transformer Model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2  # Binary classification: helmet or no-helmet
)

# Define Processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Dataset Preparation
train_csv_path = "./train_labels.csv"
test_csv_path = "./test_labels.csv"
base_dir = "./dataset/images"

# Create datasets and DataLoaders
train_dataset = HelmetDataset(train_csv_path, processor, base_dir)
test_dataset = HelmetDataset(test_csv_path, processor, base_dir)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print model summary
print(f"Model loaded with {model.config.num_labels} labels, ready for training on device: {device}")
'''


'''
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTImageProcessor

class HelmetDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor, base_dir=None):
        """
        Initialize the dataset with image paths, labels, and the feature extractor.
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of labels corresponding to the images.
            feature_extractor: The feature extractor for preprocessing.
            base_dir (str): Optional base directory to prepend to relative paths.
        """
        self.image_paths = [self.validate_path(p, base_dir) for p in image_paths]
        self.labels = labels
        self.feature_extractor = feature_extractor

        if not self.image_paths:
            raise ValueError("No valid image paths found after validation.")

    def validate_path(self, image_path, base_dir):
        """
        Validate and correct the image path.
        Args:
            image_path (str): The image file path.
            base_dir (str): Base directory for relative paths.
        Returns:
            str: Validated and corrected image path.
        """
        if base_dir and not os.path.isabs(image_path):
            image_path = os.path.join(base_dir, image_path)

        if not os.path.isfile(image_path):
            print(f"Invalid image path: {image_path}")
            return None
        return image_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a single data point.
        Args:
            idx (int): Index of the data point.
        Returns:
            tuple: Processed image tensor and label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)

# Assuming `train_image_paths`, `train_labels`, `val_image_paths`, `val_labels`, and `base_dir` are already defined
# Initialize the image processor
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Specify the base directory for relative paths
base_dir = "./dataset/train/images"

# Initialize datasets
train_data = HelmetDataset(train_image_paths, train_labels, feature_extractor, base_dir=base_dir)
val_data = HelmetDataset(val_image_paths, val_labels, feature_extractor, base_dir=base_dir)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

print("Datasets and DataLoaders initialized successfully.")
'''


'''
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HelmetDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor):
        self.image_paths = image_paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label)

# Assuming feature_extractor, train_image_paths, train_labels, val_image_paths, val_labels are already defined
train_data = HelmetDataset(train_image_paths, train_labels, feature_extractor)
val_data = HelmetDataset(val_image_paths, val_labels, feature_extractor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
'''
