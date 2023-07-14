import cv2
import numpy as np

# Load YOLOv3 model files
model_weights = 'C:/Users/suris/OneDrive/Desktop/darknet/yolov3.weights'
model_config = 'C:/Users/suris/OneDrive/Desktop/darknet/cfg/yolov3.cfg'
model_classes = 'C:/Users/suris/OneDrive/Desktop/darknet/data/coco.names'

# Load the network
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Load the class labels
classes = []
with open(model_classes, 'r') as f:
    classes = f.read().splitlines()

# Define the object of interest
object_of_interest = 'bottle'

# Access the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass through the network
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)

    # Process the detections
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == object_of_interest:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Count the objects of interest
    object_count = 0
    for i in indices:
        class_id = class_ids[i]
        if classes[class_id] == object_of_interest:
            object_count += 1

    # Draw the bounding boxes and labels
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
			1/2, color, 2)

    # Add the object count to the caption
    caption = f"Objects Detected: {object_count}"
    cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Image",frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
    for i in indices:
        i = i[0]
        box = boxes[i]
        left, top, width, height = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the frame
    cv2.imshow('Live Object Detection', frame)


"""


# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()