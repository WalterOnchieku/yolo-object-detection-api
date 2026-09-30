"""
video_detector.py

Runs YOLOv8n on a video file, draws custom bounding boxes,
and produces a widely compatible H.264 MP4 video.

Why are we using FFmpeg?

OpenCV can write video files, but the codec/container combination
can sometimes produce files that certain media players cannot play.

Therefore:

    OpenCV -> processes frames
    FFmpeg -> creates the final playable MP4
"""

from pathlib import Path
import subprocess

import cv2
from ultralytics import YOLO


# =========================================================
# STEP 1 — Define input and output paths
# =========================================================

input_video = Path(
    "input_videos/test_video.mp4"
)

# This is a temporary file.
# OpenCV will write our annotated frames here.
temporary_video = Path(
    "output/videos/annotated_temp.avi"
)

# This is the final video that we will actually use.
final_video = Path(
    "output/videos/annotated_video.mp4"
)


# =========================================================
# STEP 2 — Make sure the output directory exists
# =========================================================

final_video.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# STEP 3 — Load YOLOv8n
# =========================================================

# We load the model ONCE before processing the video.
#
# Loading it once is much more efficient than loading
# YOLO separately for every frame.
model = YOLO("yolov8n.pt")


# =========================================================
# STEP 4 — Open the input video
# =========================================================

cap = cv2.VideoCapture(
    str(input_video)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video: {input_video}"
    )


# =========================================================
# STEP 5 — Read video properties
# =========================================================

# Video width.
width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

# Video height.
height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

# Original frames per second.
fps = cap.get(
    cv2.CAP_PROP_FPS
)

# Total number of frames.
total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("\nVIDEO INFORMATION")
print("=" * 50)
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {total_frames}")
print("=" * 50)


# =========================================================
# STEP 6 — Create temporary video writer
# =========================================================

# We deliberately use AVI here.
#
# The purpose of this file is simply to store the processed
# frames before FFmpeg performs the final H.264 encoding.
fourcc = cv2.VideoWriter_fourcc(
    *"MJPG"
)

writer = cv2.VideoWriter(
    str(temporary_video),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create temporary video writer."
    )


# =========================================================
# STEP 7 — Process the video frame by frame
# =========================================================

frame_number = 0

while True:

    # -----------------------------------------------------
    # Read one frame
    # -----------------------------------------------------

    ret, frame = cap.read()

    # ret becomes False when there are no more frames.
    if not ret:
        break

    frame_number += 1

    # -----------------------------------------------------
    # Run YOLO on the current frame
    # -----------------------------------------------------

    results = model.predict(
        source=frame,
        conf=0.4,
        verbose=False
    )

    result = results[0]

    # -----------------------------------------------------
    # Process YOLO detections
    # -----------------------------------------------------

    for box in result.boxes:

        # Class ID.
        class_id = int(
            box.cls[0]
        )

        # Convert class ID to class name.
        class_name = result.names[
            class_id
        ]

        # Confidence score.
        confidence = float(
            box.conf[0]
        )

        # Bounding box coordinates.
        x1, y1, x2, y2 = [
            int(value)
            for value in box.xyxy[0]
        ]

        # -------------------------------------------------
        # Draw bounding box
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 200, 100),
            2
        )

        # -------------------------------------------------
        # Create label
        # -------------------------------------------------

        label = (
            f"{class_name} "
            f"{confidence:.0%}"
        )

        # -------------------------------------------------
        # Draw label
        # -------------------------------------------------

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 100),
            2,
            cv2.LINE_AA
        )

    # -----------------------------------------------------
    # Save the annotated frame
    # -----------------------------------------------------

    writer.write(frame)

    # -----------------------------------------------------
    # Show progress every 30 frames
    # -----------------------------------------------------

    if frame_number % 30 == 0:

        print(
            f"Processed "
            f"{frame_number}/{total_frames} frames"
        )


# =========================================================
# STEP 8 — Release OpenCV resources
# =========================================================

cap.release()
writer.release()


print("\nOpenCV processing complete.")


# =========================================================
# STEP 9 — Convert temporary AVI to H.264 MP4
# =========================================================

print("\nEncoding final MP4 with FFmpeg...")

ffmpeg_command = [
    "ffmpeg",

    # Don't ask questions if output already exists.
    "-y",

    # Input temporary video.
    "-i",
    str(temporary_video),

    # H.264 video codec.
    "-c:v",
    "libx264",

    # Good balance between quality and file size.
    "-crf",
    "23",

    # Makes the video widely compatible.
    "-pix_fmt",
    "yuv420p",

    # Output file.
    str(final_video)
]


# Run FFmpeg.
subprocess.run(
    ffmpeg_command,
    check=True
)


# =========================================================
# STEP 10 — Remove temporary video
# =========================================================

temporary_video.unlink(
    missing_ok=True
)


# =========================================================
# STEP 11 — Finished
# =========================================================

print("\nSUCCESS!")
print("=" * 50)
print(
    f"Annotated video:\n{final_video}"
)
print("=" * 50)