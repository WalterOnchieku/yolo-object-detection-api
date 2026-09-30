"""
detector.py

This module contains the computer-vision logic.

The goal is to keep YOLO and OpenCV code separate from FastAPI.
That way, the detector can later be reused by:
    - FastAPI
    - a command-line script
    - a batch-processing script
    - a video-processing script
"""

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


# Load the model once when this module is imported.
#
# This is important.
# We DON'T want to load YOLO every time someone uploads an image,
# because loading the model repeatedly would waste time and memory.
model = YOLO("yolov8n.pt")


def detect_image(
    image_path: str,
    output_path: str,
    confidence_threshold: float = 0.4
):
    """
    Run YOLO detection on one image.

    Parameters:
        image_path:
            Location of the input image.

        output_path:
            Where the annotated image should be saved.

        confidence_threshold:
            Minimum confidence required for a detection.

    Returns:
        A dictionary containing:
            - detections
            - summary
            - output path
    """

    # ---------------------------------------------------------
    # STEP 1 — Load the image
    # ---------------------------------------------------------
    #
    # OpenCV loads images as NumPy arrays.
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # ---------------------------------------------------------
    # STEP 2 — Run YOLO
    # ---------------------------------------------------------
    #
    # YOLO performs:
    #   classification -> WHAT is present?
    #   localization   -> WHERE is it?
    #
    # The result contains bounding boxes, class IDs
    # and confidence scores.
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    # This list will contain information about every
    # detected object.
    detections = []

    # ---------------------------------------------------------
    # STEP 3 — Process each detection
    # ---------------------------------------------------------
    for box in result.boxes:

        # Extract the class ID.
        class_id = int(box.cls[0])

        # Convert the class ID into a readable class name.
        class_name = result.names[class_id]

        # Extract confidence.
        confidence = float(box.conf[0])

        # YOLO gives us:
        #
        # x1, y1 = top-left corner
        # x2, y2 = bottom-right corner
        #
        # These are pixel coordinates for this image.
        x1, y1, x2, y2 = [
            int(value) for value in box.xyxy[0]
        ]

        # -----------------------------------------------------
        # STEP 4 — Draw our own bounding box
        # -----------------------------------------------------

        # Draw the object rectangle.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 200, 100),
            2
        )

        # Create the text that will appear above the box.
        label = f"{class_name} {confidence:.0%}"

        # OpenCV does not automatically create a nice
        # background behind text, so we create one ourselves.
        text_width = max(100, len(label) * 10)

        text_top = max(0, y1 - 25)

        cv2.rectangle(
            image,
            (x1, text_top),
            (x1 + text_width, y1),
            (0, 200, 100),
            -1
        )

        # Draw the class name and confidence.
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        # -----------------------------------------------------
        # STEP 5 — Store structured detection information
        # -----------------------------------------------------
        #
        # We don't just draw the result.
        # We also store the data because we need it later
        # to calculate our summary table and return JSON
        # through FastAPI.
        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            }
        )

    # ---------------------------------------------------------
    # STEP 6 — Create the summary table
    # ---------------------------------------------------------
    #
    # We convert our detection list into a pandas DataFrame.
    # This makes grouping and calculating averages easy.
    if detections:

        df = pd.DataFrame(detections)

        summary_df = (
            df.groupby("class")
            .agg(
                count=("class", "size"),
                average_confidence=("confidence", "mean")
            )
            .reset_index()
        )

        # Convert confidence from:
        #
        # 0.8734
        #
        # into:
        #
        # 87.34
        #
        # This is easier to read in our API response.
        summary_df["average_confidence"] *= 100

        summary = summary_df.to_dict(
            orient="records"
        )

    else:

        # If YOLO detects nothing, we still return
        # a valid response rather than crashing.
        summary = []

    # ---------------------------------------------------------
    # STEP 7 — Save annotated image
    # ---------------------------------------------------------

    # Make sure the output directory exists.
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        output_path,
        image
    )

    # ---------------------------------------------------------
    # STEP 8 — Return structured results
    # ---------------------------------------------------------
    return {
        "detections": detections,
        "summary": summary,
        "output_path": output_path
    }