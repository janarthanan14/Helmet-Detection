import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import ViTForImageClassification, AdamW
import os

# Custom Dataset to load images and labels from CSV file
class HelmetDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)  # Reading the CSV file
        self.image_paths = self.data['image_path'].values  # Image paths
        self.labels = self.data['label'].values  # Labels
        self.transform = transform  # Optional transformation

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # Get image path
        label = self.labels[idx]  # Get label

        # Open the image file
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)  # Apply transformation

        return image, label

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (input size for ViT)
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# Load the dataset
dataset = HelmetDataset(csv_file="helmet_labels.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load pre-trained ViT model and modify for 2-class classification (helmet/no-helmet)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)  # Optimizer
criterion = nn.CrossEntropyLoss()  # Loss function

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Ensure model is in training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU if available
        
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)  # Compute loss

        # Backward pass
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        total_loss += loss.item()  # Accumulate loss

        # Calculate accuracy
        _, preds = torch.max(outputs.logits, dim=1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_preds / total_preds
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save the trained model
model.save_pretrained("trained_helmet_vit_model")
