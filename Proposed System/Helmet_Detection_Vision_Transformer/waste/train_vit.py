import torch
from torch import nn
from torch.optim import AdamW
from transformers import ViTForImageClassification
from helmet_dataset import HelmetDataset, transform  # Assuming 'transform' is defined in helmet_dataset
from torch.utils.data import DataLoader
import io

# Load the pre-trained ViT model with 2 output classes
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)

# Move model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare the dataset and DataLoader
dataset = HelmetDataset(csv_file="helmet_labels.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Ensure model is in training mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)  # Move data to device

        optimizer.zero_grad()  # Reset gradients

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, preds = torch.max(outputs.logits, dim=1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_preds / total_preds
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save the trained model in local memory
model_local_memory = io.BytesIO()  # Create an in-memory byte stream
torch.save(model.state_dict(), model_local_memory)  # Save the model state dictionary

# Reset the pointer to the start of the byte stream for later use
model_local_memory.seek(0)

# You can now use model_local_memory as a local variable to store or pass the model
# Example: Loading the model from local memory
model_new = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
model_new.load_state_dict(torch.load(model_local_memory))  # Load from memory
model_new.to(device)
