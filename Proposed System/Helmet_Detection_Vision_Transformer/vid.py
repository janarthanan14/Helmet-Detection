import cv2
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import cvzone
from mtcnn import MTCNN
import numpy as np

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Initialize Vision Transformer (ViT) model and image processor
model_name = "./models/helmet_vit"  # Replace with your custom-trained model for helmets
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.75

# Open video file or camera feed
video_path = "Media/bike_2.mp4"  # Replace with your video file path or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces with MTCNN
    mtcnn_detections = mtcnn.detect_faces(img_rgb)

    # Prepare detections
    all_detections = []
    for detection in mtcnn_detections:
        x, y, w, h = map(int, detection['box'])  # Extract bounding box
        padding = max(10, int(min(w, h) * 0.1))  # Dynamic padding based on box size
        x = max(0, x - padding)
        y = max(0, y - padding)
        w += padding * 2
        h += padding * 2
        if w > 0 and h > 0:  # Reject invalid boxes
            all_detections.append((x, y, w, h))

    # Process all detections
    if all_detections:
        cropped_images = [
            cv2.resize(img_rgb[y:y+h, x:x+w], (224, 224), interpolation=cv2.INTER_AREA) for x, y, w, h in all_detections
        ]
        inputs = processor(images=cropped_images, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Annotate results for each detection
        for (x, y, w, h), prob in zip(all_detections, probabilities):
            predicted_class = torch.argmax(prob).item()
            confidence = prob[predicted_class].item()

            # Apply confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            if predicted_class == 1:  # Helmet
                label = "Helmet"
                color = (0, 255, 0)  # Green for Helmet
            else:  # No Helmet
                label = "No Helmet"
                color = (0, 0, 255)  # Red for No Helmet

            # Draw bounding box and label
            cvzone.cornerRect(frame, (x, y, w, h), l=10, rt=2, colorR=color)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    else:
        cv2.putText(
            frame,
            "No detections",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Display the frame with detections
    cv2.imshow("Enhanced Helmet Detection - Video", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



'''
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

# Open video file or camera feed
video_path = "Media/bike_1.mp4"  # Replace with your video file path or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces with MTCNN
    mtcnn_detections = mtcnn.detect_faces(img_rgb)

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
            cvzone.cornerRect(frame, (x, y, w, h), l=10, rt=2, colorR=color)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    else:
        cv2.putText(
            frame,
            "No detections",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Display the frame with detections
    cv2.imshow("Helmet Detection - Video", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
'''
