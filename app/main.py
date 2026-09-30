from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile

from app.detector import detect_image
from app.video_detector import process_video


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description=(
        "Object detection API using FastAPI and YOLOv8n. "
        "Supports image and video detection."
    ),
    version="1.0.0"
)


# =========================================================
# Application directories
# =========================================================

UPLOAD_DIR = Path("output/uploads")
ANNOTATED_IMAGE_DIR = Path("output/annotated")
ANNOTATED_VIDEO_DIR = Path("output/videos")


# Create directories if they don't already exist.

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATED_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ANNOTATED_VIDEO_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Health check
# =========================================================

@app.get("/health")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "ok",
        "model": "YOLOv8n",
        "image_detection": True,
        "video_detection": True
    }


# =========================================================
# Image detection endpoint
# =========================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):
    """
    Upload an image and run YOLO object detection.
    """

    # Generate a unique ID.
    unique_id = uuid.uuid4().hex

    # Remove any directory information from the filename.
    # Example: /home/user/photo.jpg becomes: photo.jpg
    original_name = Path(
        file.filename
    ).name

    
    # Create paths for the uploaded and processed files.
    input_path = (
        UPLOAD_DIR /
        f"{unique_id}_{original_name}"
    )

    output_path = (
        ANNOTATED_IMAGE_DIR /
        f"annotated_{unique_id}_{original_name}"
    )

    # Save the uploaded image.
    with input_path.open("wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Run YOLO image detection.
    result = detect_image(
        image_path=str(input_path),
        output_path=str(output_path),
        confidence_threshold=0.4
    )

    # Return the detection results as JSON.
    return {
        "filename": original_name,
        "detections": result["detections"],
        "summary": result["summary"],
        "output_file": str(output_path)
    }


# =========================================================
# Video detection endpoint
# =========================================================

@app.post("/detect-video")
async def detect_video(
    file: UploadFile = File(...)
):
    
    # Generate a unique ID for this upload.
    unique_id = uuid.uuid4().hex

    # Get a safe version of the original filename.
    original_name = Path(
        file.filename
    ).name

    # Save the uploaded video.
   
    input_path = (
        UPLOAD_DIR /
        f"{unique_id}_{original_name}"
    )

    
    # Create the name of the final annotated video.
    output_path = (
        ANNOTATED_VIDEO_DIR /
        f"annotated_{unique_id}.mp4"
    )

    # Save the uploaded video to disk.
    with input_path.open("wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Run YOLO video processing.
    result = process_video(
        input_path=str(input_path),
        output_path=str(output_path),
        confidence_threshold=0.4
    )

    # Return information about the processed video.
    return {
        "filename": original_name,
        "output_file": result["output_path"],
        "video": {
            "width": result["width"],
            "height": result["height"],
            "fps": result["fps"],
            "total_frames": result["total_frames"],
            "processed_frames": result["processed_frames"]
        }
    }
