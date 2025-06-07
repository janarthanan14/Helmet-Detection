import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw
import cv2

# Load model and processor
model_path = "./models/helmet_vit"
model = ViTForImageClassification.from_pretrained(model_path, output_hidden_states=True)
image_processor = ViTImageProcessor.from_pretrained(model_path)

def detect_helmets(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    inputs = image_processor(images=image, return_tensors="pt")

    # Inference
    outputs = model(**inputs)

    # Check if hidden_states is available
    if outputs.hidden_states is None:
        raise ValueError("Hidden states are not available. Check the model configuration.")

    # Replace this with your custom bounding box extraction logic
    # Example: Mock bounding box logic for demo
    results = [
        {"bbox": [50, 50, 150, 150], "label": "With Helmet", "score": 0.94},
        {"bbox": [200, 100, 300, 250], "label": "With Helmet", "score": 0.85},
    ]

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

    try:
        # Detect helmets
        predictions = detect_helmets(input_image_path)

        # Draw bounding boxes
        output_image = draw_bounding_boxes(input_image_path, predictions)

        # Save and display output
        cv2.imwrite(output_image_path, output_image)
        cv2.imshow("Helmet Detection", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error: {e}")
