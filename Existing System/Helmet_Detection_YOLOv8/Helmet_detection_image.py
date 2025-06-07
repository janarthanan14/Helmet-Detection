import cv2 # type: ignore
import math
import cvzone # type: ignore
from ultralytics import YOLO                 # type: ignore
# Load YOLO model with custom weights
yolo_model = YOLO("Weights/best.pt")
input = py 02 d 1224 # type: ignore
# Define class names
class_labels = ['With Helmet', 'Without Helmet']    
    # Load the image
image_path = "Media/riders_1.jpg"
img = cv2.imread(image_path)
    # Perform object detection
results = yolo_model(img)
    # Loop through the detections and draw bounding boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h))
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
            # Display the image with detections
cv2.imshow("Image", img)

# Close window when 'q' button is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cv2.waitKey(1)
