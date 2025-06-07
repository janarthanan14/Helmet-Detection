import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, get_scheduler
from torch.utils.data import DataLoader
from PIL import Image

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

# 6. Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
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
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler

# Define optimizer and scheduler
optimizer = Adam(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

num_training_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
'''
