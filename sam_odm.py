import numpy as np
import onnxruntime as rt
from PIL import Image

# Define a mapping from class IDs to class names (modify as needed)
class_names = {
    0: 'unknown', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    # ... add more class names as needed
}

# Load the ONNX model
sess = rt.InferenceSession("model.onnx")

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
    input_tensor = np.array(frame).astype('float32')
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run object detection
    boxes, scores, classes = sess.run(None, {sess.get_inputs()[0].name: input_tensor})

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
