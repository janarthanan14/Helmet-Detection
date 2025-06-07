import os
import glob
import pandas as pd

# Paths to your YOLO annotations and image directories
yolo_annotations_dir = "./dataset/valid/labels"
image_dir = "./dataset/valid/images"

# Create a list to store image paths and labels
data = []

# Loop through each image in the dataset
for img_path in glob.glob(os.path.join(image_dir, "*.jpg")):  # or *.png, depending on your images
    # Corresponding annotation file for the image
    annotation_file = os.path.join(yolo_annotations_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
    
    # Check if the annotation file exists
    if os.path.exists(annotation_file):
        # Read the annotation file
        with open(annotation_file, "r") as f:
            annotations = f.readlines()

        # Check if there is any helmet in the annotations (assuming class 0 is for helmet)
        helmet_found = any([line.split()[0] == "0" for line in annotations])

        # Assign label based on whether helmet is found in the image
        label = 1 if helmet_found else 0
        data.append([img_path, label])
    else:
        # If no annotations, assume no helmet in the image
        data.append([img_path, 0])

# Convert to DataFrame for easy export
df = pd.DataFrame(data, columns=["image_path", "label"])

# Save as CSV
df.to_csv("valid_labels.csv", index=False)

print("Conversion complete!")
