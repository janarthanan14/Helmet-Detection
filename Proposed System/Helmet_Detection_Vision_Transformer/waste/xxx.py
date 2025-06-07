import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Step 1: Custom Dataset Class
class HelmetDataset(Dataset):
    def __init__(self, image_paths, label_paths, feature_extractor, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Parse label file
        with open(label_path, "r") as file:
            labels = file.readlines()

        # Binary classification: if there's at least one '1', assign the label as 1 (Helmet present)
        binary_label = 1 if any(int(line.split()[0]) == 1 for line in labels) else 0

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(),
            "label": torch.tensor(binary_label, dtype=torch.long),
        }

# Step 2: Data Preparation
def prepare_data(data_dir, feature_extractor, test_size=0.2):
    image_paths = []
    label_paths = []

    # Assume images are in "images/" and labels are in "labels/" subdirectories
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")

    for file in os.listdir(images_dir):
        if file.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(images_dir, file)
            label_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")

            if os.path.exists(label_path):
                image_paths.append(image_path)
                label_paths.append(label_path)

    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, label_paths, test_size=test_size, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    train_dataset = HelmetDataset(train_images, train_labels, feature_extractor, transform)
    val_dataset = HelmetDataset(val_images, val_labels, feature_extractor, transform)

    return train_dataset, val_dataset

# Step 3: Model Training
def train_model(model, train_loader, val_loader, device, epochs=5):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=inputs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)

                outputs = model(pixel_values=inputs).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        print(f"Validation Loss: {val_loss / len(val_loader)}, Accuracy: {val_accuracy:.2f}")

# Step 4: Main Function
def main(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2)

    train_dataset, val_dataset = prepare_data(data_dir, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    train_model(model, train_loader, val_loader, device)
    model.save_pretrained("./helmet_detection_model")

# Run the script
if __name__ == "__main__":
    data_directory = "./data"  # Replace with your dataset directory
    main(data_directory)
