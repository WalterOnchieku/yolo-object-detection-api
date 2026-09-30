"""
test_batch.py

Runs YOLOv8n against all images in input_images/.

This script is our batch-testing layer.
It allows us to prove that the detector works on
multiple images rather than just one.
"""

from pathlib import Path

from app.detector import detect_image


# ---------------------------------------------------------
# STEP 1 — Find our test images
# ---------------------------------------------------------

input_dir = Path("input_images")
output_dir = Path("output/annotated")

# We support the most common image formats.
image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

images = [
    path
    for path in input_dir.iterdir()
    if path.suffix.lower() in image_extensions
]

# Sort them so the processing order is predictable.
images.sort()


# ---------------------------------------------------------
# STEP 2 — Make sure we have enough test images
# ---------------------------------------------------------

if len(images) < 5:

    raise ValueError(
        f"We need at least 5 test images. "
        f"Currently found {len(images)}."
    )


print(f"Found {len(images)} test images.")


# ---------------------------------------------------------
# STEP 3 — Process every image
# ---------------------------------------------------------

for image_path in images:

    print("\n" + "=" * 70)
    print(f"Processing: {image_path.name}")
    print("=" * 70)

    output_path = (
        output_dir /
        f"annotated_{image_path.name}"
    )

    result = detect_image(
        image_path=str(image_path),
        output_path=str(output_path),
        confidence_threshold=0.4
    )

    # -----------------------------------------------------
    # STEP 4 — Print the required summary table
    # -----------------------------------------------------

    print(
        f"{'Class':<20}"
        f"{'Count':<10}"
        f"{'Avg Confidence':<20}"
    )

    print("-" * 50)

    for item in result["summary"]:

        print(
            f"{item['class']:<20}"
            f"{item['count']:<10}"
            f"{item['average_confidence']:.2f}%"
        )

    print(
        f"\nSaved: {result['output_path']}"
    )