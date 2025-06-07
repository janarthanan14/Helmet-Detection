def count_words(text):
    words = text.split()  # Split the text into words based on whitespace
    return len(words)     # Return the number of words

# Example usage
text = """Ensuring road safety, particularly for motorcyclists, is a critical global concern. One effective measure to enhance safety for riders is the use of helmets, which significantly reduces the risk of severe head injuries in accidents. However, enforcing helmet usage can be challenging due to limited resources for manual monitoring and enforcement. This project aims to develop an automated helmet detection system using an improved version of the YOLO V8 (You Only Look Once) object detection algorithm, tailored for real-time helmet recognition. By optimizing YOLO V8, the system achieves high accuracy in detecting whether a motorcyclist is wearing a helmet, even in varied and challenging conditions, such as poor lighting, occlusion, or varying helmet colors.

The project involves the design, training, and deployment of a neural network model enhanced with specific improvements over the base YOLO V8 architecture to boost its robustness and speed. The system is designed to be integrated into real-world applications, such as surveillance cameras at traffic intersections, where it can continuously monitor and identify helmet usage among motorcyclists. The proposed solution addresses the limitations of previous models by improving detection rates, reducing false positives, and enhancing computational efficiency, making it suitable for large-scale deployment.

The results demonstrate that the improved YOLO V8 model achieves a significant increase in both detection accuracy and processing speed compared to previous versions, proving it to be an effective tool for enforcing helmet compliance in real time. This project contributes to the field of traffic safety automation and has the potential to assist traffic authorities in ensuring compliance with helmet-wearing regulations, ultimately reducing fatalities and injuries among motorcyclists."""
word_count = count_words(text)
print("Number of words:", word_count)
