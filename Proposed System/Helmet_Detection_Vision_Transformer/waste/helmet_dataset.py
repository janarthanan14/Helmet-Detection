import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HelmetDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and labels.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the image and label at the specified index.
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the corresponding label.
        """
        img_path = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1])  # Ensure the label is an integer

        # Check if the image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Open the image and convert it to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, label

# Define the transformation (resize, normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ViT input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def main():
    # Path to the CSV file
    csv_file = "valid_labels.csv"

    # Ensure the CSV file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    # Initialize the dataset
    dataset = HelmetDataset(csv_file=csv_file, transform=transform)

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Iterate through the DataLoader
    print("Iterating through the DataLoader...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images batch shape: {images.shape}")  # Example: torch.Size([32, 3, 224, 224])
        print(f"  Labels batch shape: {labels.shape}")  # Example: torch.Size([32])
        print(f"  Labels: {labels.tolist()}")

if __name__ == "__main__":
    main()
