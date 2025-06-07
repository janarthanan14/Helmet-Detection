import torch
from torch import nn
from torch.optim import AdamW
from transformers import ViTForImageClassification
from helmet_dataset import HelmetDataset
from torch.utils.data import DataLoader

#Load pre-trained ViT model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)

#Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Prepare the dataset
from helmet_dataset import transform
dataset = HelmetDataset(csv_file="helmet_labels.csv", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

#Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

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

#Save the trained model (optional)
model.save_pretrained("trained_helmet_vit_model")
