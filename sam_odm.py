import numpy as np
import tensorflow as tf
from PIL import Image

# Define a mapping from class IDs to class names (modify as needed)
class_names = {
    0: 'unknown', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    # ... add more class names as needed
}

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the camera
cap = Image.open(0)

while True:
    # Read a frame from the camera
    frame = cap.read()

    # Check if frame was read successfully
    if frame is None:
        print("Error reading frame from camera")
        break

    # Preprocess the frame for the model
    frame = frame.convert('RGB')
    frame = frame.resize((640, 480))
    input_tensor = tf.convert_to_tensor(np.array(frame), dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run inference
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])  # Confidence of detected objects

    # Draw bounding boxes and labels on the frame
    for box, score, class_id in zip(boxes[0], scores[0], classes[0]):
        if score > 0.5:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * frame.width), int(y1 * frame.height), int(x2 * frame.width), int(y2 * frame.height)
            # Drawing operations are not supported in Pillow, you might need to use other libraries if you want to draw on the image

    # Display the frame
    frame.show()

    # Press 'q' to exit
    if input() == 'q':
        break

# Close all windows
cap.close()
