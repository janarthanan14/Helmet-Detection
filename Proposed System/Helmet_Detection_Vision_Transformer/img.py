import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import cvzone
from mtcnn import MTCNN
import numpy as np

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Initialize Vision Transformer (ViT) model and image processor
model_name = "./models/helmet_vit"  # Replace with a custom-trained model for helmets
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load and preprocess the image
#"Media/riders_5.jpg" 
image_path = "Media/riders_3.jpg"  # Replace with your image path
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect faces with MTCNN
mtcnn_detections = mtcnn.detect_faces(img)

# Prepare detections
all_detections = []
for detection in mtcnn_detections:
    x, y, w, h = map(int, detection['box'])  # Extract bounding box
    padding = 20  # Add padding for better helmet detection
    x = max(0, x - padding)
    y = max(0, y - padding)
    w += padding * 2
    h += padding * 2
    all_detections.append((x, y, w, h))

# Process all detections (none are ignored)
if all_detections:
    cropped_images = [cv2.resize(img_rgb[y:y+h, x:x+w], (224, 224)) for x, y, w, h in all_detections]
    inputs = processor(images=cropped_images, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Annotate results for each detection
    for (x, y, w, h), prob in zip(all_detections, probabilities):
        predicted_class = torch.argmax(prob).item()
        confidence = prob[predicted_class].item()

        if predicted_class == 1:  # Helmet
            label = "Helmet"
            color = (0, 255, 0)  # Green for Helmet
        else:  # No Helmet
            label = "No Helmet"
            color = (0, 0, 255)  # Red for No Helmet

        # Draw bounding box and label
        cvzone.cornerRect(img, (x, y, w, h), l=10, rt=2, colorR=color)
        cv2.putText(
            img,
            f"{label} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
else:
    print("No detections found.")

# Display the image with detections
cv2.imshow("Helmet Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
