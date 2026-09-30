from app.detector import detect_image


# We are testing the detector independently of FastAPI.
# This makes debugging easier because we know whether
# a problem belongs to YOLO/OpenCV or to the API layer.

result = detect_image(
    image_path="input_images/image1.jpg",
    output_path="output/annotated/image1.jpg",
    confidence_threshold=0.4
)


# Print the raw detections.
print("\nDETECTIONS")
print("=" * 60)

for detection in result["detections"]:
    print(
        detection["class"],
        f"{detection['confidence']:.2%}",
        detection["box"]
    )


# Print our required summary table.
print("\nSUMMARY")
print("=" * 60)

print(
    f"{'Class':<20}"
    f"{'Count':<10}"
    f"{'Avg Confidence':<20}"
)

print("-" * 60)

for item in result["summary"]:
    print(
        f"{item['class']:<20}"
        f"{item['count']:<10}"
        f"{item['average_confidence']:.2f}%"
    )

print("\nSaved to:")
print(result["output_path"])