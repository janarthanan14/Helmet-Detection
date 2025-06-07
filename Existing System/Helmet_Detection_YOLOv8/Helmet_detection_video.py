import cv2
import math
import cvzone
from ultralytics import YOLO
import os

# Initialize video capture
video_path = "Media/sample.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file '{video_path}' not found.")

cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
weights_path = "Weights/best5.pt"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights file '{weights_path}' not found.")
    
model = YOLO(weights_path)

# Define class names
classNames = ['With Helmet', 'Without Helmet']
frame = 0

while True:
    success, img = cap.read()
    if not success:
        print("End of video or error reading frame.")
        break

    # YOLO model inference
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Display confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Display frame
    frame += 1
    print(f"Processed frame: {frame}")
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
