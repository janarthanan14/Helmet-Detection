# Save the model
model.save_pretrained("./models/helmet_vit")
feature_extractor.save_pretrained("./models/helmet_vit")

# Load for inference
from transformers import pipeline

helmet_detector = pipeline("image-classification", model="./models/helmet_vit")
result = helmet_detector("./test_image.jpg")
print(result)
