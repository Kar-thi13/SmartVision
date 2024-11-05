import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import object_detection
tf.disable_v2_behavior()  # Disable TensorFlow 2.x features
from object_detection.utils import label_map_util

# List of class labels for the object detection model
classes = ["background", "person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
           "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
           "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Colors for labels
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open the webcam
cam = cv.VideoCapture(1)

# Model files
pb = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'

# Check if files exist
if not os.path.isfile(pb):
    raise FileNotFoundError(f"File '{pb}' does not exist.")
if not os.path.isfile(pbt):
    raise FileNotFoundError(f"File '{pbt}' does not exist.")

# Load the neural network
cvNet = cv.dnn.readNetFromTensorflow(pb, pbt)

while True:
    # Read in the frame from the webcam
    ret_val, img = cam.read()
    if not ret_val:
        print("Failed to capture image")
        break

    rows = img.shape[0]
    cols = img.shape[1]

    # Prepare the image for object detection
    blob = cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False)
    cvNet.setInput(blob)

    # Run object detection
    cvOut = cvNet.forward()

    # Process each detected object
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            idx = int(detection[1])  # Prediction class index

            # Check for specific utensils
            if classes[idx] in ['fork', 'spoon', 'knife']:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows

                # Draw bounding box
                cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

                # Draw label
                label = "{}: {:.2f}%".format(classes[idx], score * 100)
                y = top - 15 if top - 15 > 15 else top + 15
                cv.putText(img, label, (int(left), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    # Display the frame
    cv.imshow('my webcam', img)

    # Exit on ESC key
    if cv.waitKey(1) == 27:
        break

# Release the webcam and close OpenCV windows
cam.release()
cv.destroyAllWindows()
