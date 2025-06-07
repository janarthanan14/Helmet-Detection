import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw, ImageFont
import cv2

# Load model and processor
model_path = "./models/helmet_vit"
model = ViTForImageClassification.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained(model_path)

def detect_helmets(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    inputs = image_processor(images=image, return_tensors="pt")
    
    # Inference
    outputs = model(**inputs)
    logits = outputs.logits  # Class logits
    bounding_boxes = outputs.hidden_states[-1]  # Extract bounding box predictions from the last layer (example)

    # Convert logits to class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    scores, indices = probabilities.max(dim=1)

    # Filter predictions based on confidence threshold
    confidence_threshold = 0.5
    results = []
    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            # Example: Map bounding boxes to original image size
            box = bounding_boxes[i].tolist()  # Replace this with real bounding box logic
            x_min, y_min, x_max, y_max = (
                int(box[0] * original_width),
                int(box[1] * original_height),
                int(box[2] * original_width),
                int(box[3] * original_height),
            )
            results.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "label": f"With Helmet {scores[i]:.2f}",
                "score": scores[i].item(),
            })

    return results

def draw_bounding_boxes(image_path, predictions):
    # Open the image
    image = cv2.imread(image_path)
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        label = pred['label']

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Draw text
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            (0, 255, 0),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
        )

    return image

# Main execution
if __name__ == "__main__":
    input_image_path = "./Media/riders_1.jpg"
    output_image_path = "./output/detected_riders.jpg"

    # Detect helmets
    predictions = detect_helmets(input_image_path)

    # Draw bounding boxes
    output_image = draw_bounding_boxes(input_image_path, predictions)

    # Save and display output
    cv2.imwrite(output_image_path, output_image)
    cv2.imshow("Helmet Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



'''
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw, ImageFont
import cv2
import cvzone

# Load model and processor
model_path = "./models/helmet_vit"
model = ViTForImageClassification.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained(model_path)

def detect_helmets(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Post-process predictions
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    scores, indices = probabilities.max(dim=1)

    # Example: Replace with real bounding box logic
    results = [
        {"bbox": [50, 50, 150, 150], "label": "With Helmet", "score": 0.94},
        {"bbox": [200, 100, 300, 250], "label": "With Helmet", "score": 0.85},
    ]

    return results

def draw_bounding_boxes(image_path, predictions):
    image = cv2.imread(image_path)
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        label = f"{pred['label']} {pred['score']:.2f}"

        # Draw bounding box and text
        cvzone.cornerRect(image, (x1, y1, x2-x1, y2-y1), colorR=(0, 255, 0))
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Main
if __name__ == "__main__":
    input_image_path = "./Media/riders_1.jpg"
    output_image_path = "./output/d1.jpg"

    predictions = detect_helmets(input_image_path)
    output_image = draw_bounding_boxes(input_image_path, predictions)

    # Save and display output
    cv2.imwrite(output_image_path, output_image)
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''


'''
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw, ImageFont
import os
import math
import cv2
import numpy as np

# Load pre-trained model and image processor
model_path = "./models/helmet_vit"
model = ViTForImageClassification.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained(model_path)

# Draw bounding boxes with OpenCV for better visualization
def draw_bounding_boxes_cv(image_path, predictions):
    image = cv2.imread(image_path)
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        label = pred['label']
        score = pred['score']
        conf = math.ceil(score * 100) / 100

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"

        # Put text above the bounding box
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            (0, 255, 0),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
    return image

# Dummy detection logic to simulate bounding boxes and predictions
def detect_helmets(image_path):
    # Preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    # Predict
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    scores, indices = torch.max(probabilities, dim=-1)

    # Simulated results (use actual detection logic for real implementation)
    results = [
        {"bbox": [50, 50, 150, 150], "label": "With Helmet", "score": 0.94},
        {"bbox": [200, 100, 300, 250], "label": "With Helmet", "score": 0.85},
    ]

    return results

# Main script
if __name__ == "__main__":
    # Path to the input image
    input_image_path = "./Media/riders_1.jpg"  # Replace with actual path
    output_image_path = "./output/d1.jpg"

    # Perform helmet detection
    predictions = detect_helmets(input_image_path)

    # Draw bounding boxes using OpenCV
    output_image = draw_bounding_boxes_cv(input_image_path, predictions)

    # Save and display the output image
    cv2.imwrite(output_image_path, output_image)
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Output image saved to {output_image_path}")
'''


'''
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw, ImageFont
import os

# Load pre-trained model and image processor
model_path = "./models/helmet_vit"
model = ViTForImageClassification.from_pretrained(model_path)
image_processor = ViTImageProcessor.from_pretrained(model_path)

# Function to generate bounding boxes with predictions
def draw_bounding_boxes(image_path, predictions):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for pred in predictions:
        box, label, score = pred['bbox'], pred['label'], pred['score']
        draw.rectangle(box, outline="green", width=3)
        text = f"{label} {score:.2f}"
        
        # Get text size using textbbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_position = (box[0], box[1] - text_height)

        # Draw text background
        draw.rectangle([text_position, (box[0] + text_width, box[1])], fill="green")
        draw.text(text_position, text, fill="white", font=font)

    return image

# Dummy detection logic to simulate bounding boxes and predictions
def detect_helmets(image_path):
    # Preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    # Predict
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    scores, indices = torch.max(probabilities, dim=-1)

    # Simulated results (use actual detection logic for real implementation)
    results = [
        {"bbox": [50, 50, 150, 150], "label": "With Helmet", "score": 0.94},
        {"bbox": [200, 100, 300, 250], "label": "With Helmet", "score": 0.85},
    ]

    return results

# Main script
if __name__ == "__main__":
    # Path to the input image
    input_image_path = "./Media/riders_1.jpg"  # Replace with actual path
    output_image_path = "./output/d1.jpg"

    # Perform helmet detection
    predictions = detect_helmets(input_image_path)

    # Draw bounding boxes and save the output
    output_image = draw_bounding_boxes(input_image_path, predictions)
    output_image.save(output_image_path)
    output_image.show()

    print(f"Output image saved to {output_image_path}")
'''


'''
from flask import Flask, request, jsonify, render_template, send_file
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import os
import io

app = Flask(__name__)

# Load pre-trained Vision Transformer (DETR) model and processor
model_path = "./models/helmet_vit"  # Pre-trained model from Hugging Face
processor = DetrImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path)

# Create upload folder
UPLOAD_FOLDER = "./Media"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded image
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Load the image
    image = Image.open(file_path).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Perform object detection
    outputs = model(**inputs)
    logits = outputs.logits
    boxes = outputs.pred_boxes

    # Filter results based on confidence threshold
    threshold = 0.7
    scores = logits.softmax(-1)[0, :, :-1].max(-1).values
    keep = scores > threshold
    boxes = boxes[0, keep].cpu().detach().numpy()
    labels = logits.softmax(-1)[0, keep].argmax(-1).cpu().detach().numpy()
    scores = scores[keep].cpu().detach().numpy()

    # Annotate the image with bounding boxes and labels
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Update with a valid font path on your system
    font = ImageFont.truetype(font_path, size=20)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box * torch.tensor([image.width, image.height, image.width, image.height]).numpy()
        label_name = processor.id2label[label]
        label_text = f"{label_name} {score:.2f}"

        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="lime", width=3)

        # Draw label
        draw.text((x1, y1 - 20), label_text, fill="lime", font=font)

    # Save the annotated image to a BytesIO object
    img_io = io.BytesIO()
    image.save(img_io, "JPEG")
    img_io.seek(0)

    # Delete the uploaded image to save space
    os.remove(file_path)

    # Send the annotated image back to the client
    return send_file(img_io, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
'''
