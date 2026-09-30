from ultralytics import YOLO


# We start with the pretrained YOLOv8 Nano model.
# At this stage we are deliberately keeping the test simple:
# first prove that YOLO can load and make a prediction,
# then we will build our own application around it.
model = YOLO("yolov8n.pt")


# Run YOLO against one image.
# Replace this filename with an image that exists in input_images/.
results = model.predict(
    source="input_images/image1.jpg",
    conf=0.4
)


# YOLO returns a list of Results objects.
# We are using the first result because we supplied one image.
result = results[0]


# result.boxes contains all detected objects.
print(f"Detected {len(result.boxes)} objects")


# Examine every detection.
for box in result.boxes:

    # YOLO's class ID is stored as a tensor.
    # int(...) converts it into a normal Python integer.
    class_id = int(box.cls[0])

    # Get the human-readable class name.
    class_name = result.names[class_id]

    # Confidence tells us how strongly the model
    # supports this detection.
    confidence = float(box.conf[0])

    print(
        f"{class_name}: "
        f"{confidence:.2%}"
    )