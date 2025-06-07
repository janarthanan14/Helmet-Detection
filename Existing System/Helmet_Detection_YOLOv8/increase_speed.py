import cv2
import math
import cvzone
from ultralytics import YOLO
import os

# Initialize video capture
video_path = "Media/sample.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
model = YOLO("Weights/best5.pt")

# Define class names
classNames = ['With Helmet', 'Without Helmet']

# Set frame skip parameter and initialize variables
frame_skip = 2  # Process every 2nd frame
frame = 0
detection_cache = []  # Stores detection results for longer visibility
cache_duration = 5  # Keep detections visible for 5 frames

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from video.")
        break

    if frame % frame_skip == 0:
        # Run detection on selected frames only
        results = model(img, stream=True)
        detection_cache = []  # Clear cache for fresh detections

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Store the detection with its bounding box and class for the cache duration
                detection_cache.append((x1, y1, w, h, classNames[cls], conf, cache_duration))

    # Draw detections from cache
    for i, (x1, y1, w, h, label, conf, duration) in enumerate(detection_cache):
        if duration > 0:
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            # Decrement the duration of this cached detection
            detection_cache[i] = (x1, y1, w, h, label, conf, duration - 1)

    # Check if a specific file exists (optional)
    print(os.path.exists("riders_1.jpg"))
    print("Frame:", frame)

    # Show the frame
    cv2.imshow("Image", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame counter
    frame += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
