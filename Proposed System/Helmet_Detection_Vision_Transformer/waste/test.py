import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# Define the transformation used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 (required input size for ViT)
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as ImageNet
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained("trained_helmet_vit_model")
model.to(device)  # Move model to GPU if available
model.eval()  # Set the model to evaluation mode

# Function to predict the class of an image
def predict_image(image_path):
    # Open the image and apply transformations
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Forward pass through the model to get predictions
    with torch.no_grad():  # No gradient calculation needed for inference
        outputs = model(image)
    
    # Get predicted class (0 or 1)
    _, predicted_class = torch.max(outputs.logits, dim=1)
    
    # Interpret the prediction (0 for no helmet, 1 for helmet)
    if predicted_class.item() == 0:
        return "No Helmet"
    else:
        return "Helmet"

# Function to test all images in the specified directory
def test_images_in_directory(image_dir):
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        
        # Ensure that the file is an image (you can expand this to check file types)
        if image_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f"Testing image: {image_name}")
            prediction = predict_image(image_path)
            print(f"Prediction for {image_name}: {prediction}")
        else:
            print(f"Skipping non-image file: {image_name}")

# Example of testing all images in the directory
image_dir = "./more-preprocessing-yolov8/test/images"  # Path to your image directory
test_images_in_directory(image_dir)
