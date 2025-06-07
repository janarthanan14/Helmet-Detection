import cv2
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import math

# Load pre-trained Vision Transformer model
model_path = "./models/helmet_vit"  # Path to your Vision Transformer model
helmet_detector = pipeline("image-classification", model=model_path)

# Define class names
class_labels = ['With Helmet', 'Without Helmet']    

# Load the image using OpenCV
image_path = "Media/riders_6.jpg"  # Update with your image path
img = cv2.imread(image_path)

# Convert to RGB for Vision Transformer (ViT) input
image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Perform prediction using Vision Transformer
result = helmet_detector(image)

# Get the predicted label and confidence
predicted_label = result[0]['label']
confidence = result[0]['score']

# Annotate the image with prediction result
label = f"{predicted_label}: {confidence:.2f}"

# Set the font for annotation
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
font_color = (0, 255, 0)  # Green text
font_thickness = 2

# Add the label to the image at the top-left corner
cv2.putText(img, label, (10, 40), font, font_size, font_color, font_thickness)

# Display the image with annotation using OpenCV
cv2.imshow("Predicted Image", img)

# Close window when 'q' key is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
