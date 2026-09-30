# YOLOv8 Object Detection API

A computer vision application built with **YOLOv8n, OpenCV, FFmpeg, and FastAPI** for detecting objects in images and videos.

The project demonstrates how a pretrained YOLO object-detection model can be integrated into a reusable Python application and exposed through a REST API.

---

## Features

* YOLOv8 Nano (`yolov8n`) object detection
* Image object detection
* Video object detection
* Custom bounding boxes
* Class names and confidence scores
* Detection summary for images
* REST API built with FastAPI
* Interactive Swagger API documentation
* OpenCV-based video frame processing
* FFmpeg-based H.264 MP4 encoding
* Support for multiple image formats
* Reusable detection modules

---

## Project Architecture

```text
yolo/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── detector.py
│   └── video_detector.py
│
├── input_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   ├── image4.jpg
│   └── image5.jpg
│
├── input_videos/
│   └── test_video.mp4
│
├── output/
│   ├── annotated/
│   ├── videos/
│   └── uploads/
│
├── test_yolo.py
├── test_detector.py
├── test_batch.py
├── requirements.txt
├── yolov8n.pt
└── README.md
```

### Application flow

```text
                    FastAPI
                       │
                    main.py
                  /         \
                 /           \
            /detect       /detect-video
               │               │
               ▼               ▼
          detector.py    video_detector.py
               │               │
               ▼               ▼
             YOLO             YOLO
               │               │
               ▼               ▼
             Image           Video
```

---

## Technologies Used

| Technology  | Purpose                          |
| ----------- | -------------------------------- |
| Python      | Application programming language |
| YOLOv8      | Object detection                 |
| Ultralytics | YOLO implementation              |
| OpenCV      | Image and video processing       |
| FFmpeg      | H.264 video encoding             |
| FastAPI     | REST API                         |
| Uvicorn     | ASGI server                      |
| Pandas      | Detection summary processing     |

---

## How YOLO Object Detection Works

YOLO stands for **You Only Look Once**.

Instead of processing an image separately for classification and localization, YOLO performs object detection in a single inference process.

For each detected object, the model provides:

* Class ID
* Class name
* Confidence score
* Bounding-box coordinates

For example:

```text
person    94%
car       87%
bus       91%
```

A bounding box is represented using:

```text
[x1, y1, x2, y2]
```

where:

```text
(x1, y1) → top-left corner
(x2, y2) → bottom-right corner
```

---

## YOLOv8n

This project uses:

```text
yolov8n.pt
```

The `n` stands for **Nano**.

YOLOv8 Nano is a relatively small model designed for fast inference and experimentation.

The model is pretrained on the **COCO dataset**, which contains 80 object categories.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/yolo-object-detection-api.git
```

Move into the project:

```bash
cd yolo-object-detection-api
```

---

## 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv venv
```

Windows:

```bash
python -m venv venv
```

Activate the environment.

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install FFmpeg

FFmpeg is required for converting the processed video into a widely compatible H.264 MP4.

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify the installation:

```bash
ffmpeg -version
```

---

# Running Image Detection

The project includes a batch-processing script for testing multiple images.

Place your test images inside:

```text
input_images/
```

Run:

```bash
python test_batch.py
```

The script processes the images using YOLOv8n and saves the annotated results to:

```text
output/annotated/
```

The terminal displays a summary similar to:

```text
======================================================================
Processing: image1.jpg
======================================================================

Class               Count     Avg Confidence
--------------------------------------------------
person              3         91.42%
car                 2         87.31%

Saved: output/annotated/annotated_image1.jpg
```

---

# Image Detection API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

## Swagger Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test the endpoints directly from the Swagger interface.

---

# API Endpoints

## GET `/health`

Checks whether the API is running.

Example response:

```json
{
    "status": "ok",
    "model": "YOLOv8n",
    "image_detection": true,
    "video_detection": true
}
```

---

## POST `/detect`

Uploads an image and performs object detection.

The endpoint:

1. Receives the uploaded image.
2. Saves the image temporarily.
3. Runs YOLO inference.
4. Extracts detected objects.
5. Draws bounding boxes.
6. Creates a detection summary.
7. Saves the annotated image.
8. Returns the detection results as JSON.

Example response:

```json
{
    "filename": "image1.jpg",
    "detections": [
        {
            "class": "person",
            "confidence": 0.94,
            "box": [
                120,
                80,
                350,
                500
            ]
        }
    ],
    "summary": [
        {
            "class": "person",
            "count": 1,
            "average_confidence": 94.0
        }
    ],
    "output_file": "output/annotated/annotated_abc123_image1.jpg"
}
```

---

# Video Detection

The project can also process video files.

Place a test video inside:

```text
input_videos/
```

The standalone video-processing functionality can be tested through the reusable `process_video()` function in:

```text
app/video_detector.py
```

The FastAPI application exposes the same functionality through:

```text
POST /detect-video
```

---

## POST `/detect-video`

Uploads a video and performs YOLO object detection on every frame.

The processing pipeline is:

```text
Uploaded Video
      │
      ▼
   OpenCV
      │
      ▼
Read individual frames
      │
      ▼
    YOLOv8
      │
      ▼
Detect objects
      │
      ▼
Draw bounding boxes
      │
      ▼
Temporary AVI
      │
      ▼
    FFmpeg
      │
      ▼
H.264 MP4
```

Example response:

```json
{
    "filename": "test_video.mp4",
    "output_file": "output/videos/annotated_abc123.mp4",
    "video": {
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
        "total_frames": 264,
        "processed_frames": 264
    }
}
```

The final annotated video is saved in:

```text
output/videos/
```

---

# Why FFmpeg Is Used

OpenCV can read and write video files, but codec support can vary depending on the environment.

This project therefore uses two stages.

### Stage 1 — OpenCV

OpenCV processes individual video frames and temporarily stores them as an AVI file using the MJPG codec.

### Stage 2 — FFmpeg

FFmpeg converts the temporary video into an H.264 MP4:

```text
MJPG AVI
   │
   ▼
FFmpeg
   │
   ▼
H.264 MP4
```

The final video uses:

```text
libx264
yuv420p
```

which provides broad compatibility with common media players and browsers.

---

# Confidence Threshold

The detector currently uses:

```python
confidence_threshold=0.4
```

This means detections below 40% confidence are filtered out.

For example:

```text
0.95 → accepted
0.82 → accepted
0.51 → accepted
0.38 → rejected
```

The confidence score represents the model's confidence in its prediction. It should not be interpreted as a guaranteed probability that the prediction is correct.

The threshold can be changed when calling the detection functions.

---

# Detection Summary

For image detection, the project produces a summary containing:

```text
Class
Count
Average Confidence
```

For example:

```text
Class               Count     Avg Confidence
--------------------------------------------------
person              4         92.13%
car                 3         86.47%
bicycle             1         79.21%
```

The count represents the number of detections in that image.

For video, the current project does **not** treat detections across frames as unique objects.

For example, if the same car appears in 100 frames, that does not mean 100 different cars were detected.

Unique-object counting across video requires object tracking.

---

# Testing

The project includes several scripts for testing different parts of the system.

### Basic YOLO test

```bash
python test_yolo.py
```

### Detector module test

```bash
python test_detector.py
```

### Batch image detection

```bash
python test_batch.py
```

### FastAPI application

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# Output Files

Generated image results are stored in:

```text
output/annotated/
```

Generated video results are stored in:

```text
output/videos/
```

Uploaded API files are stored temporarily in:

```text
output/uploads/
```

---

# Future Improvements

Possible extensions for this project include:

* Object tracking using YOLO tracking
* Unique-object counting
* Webcam detection
* Real-time video streaming
* Detection history
* Database storage
* Authentication and authorization
* File-size and file-type validation
* Pydantic API response models
* Background processing for large videos
* Docker containerization
* Automated tests
* Frontend interface
* Model performance benchmarking
* Custom YOLO model training
* Deployment to a cloud platform

---

# Learning Objectives

This project demonstrates practical experience with:

* Computer vision
* Object detection
* YOLO inference
* Bounding-box processing
* Confidence thresholds
* OpenCV
* Video processing
* FFmpeg
* REST API development
* FastAPI
* Python modular application design
* File uploads
* API testing
* Model integration

---

# Project Status

The current implementation supports:

```text
✓ YOLOv8n object detection
✓ Image detection
✓ Custom bounding boxes
✓ Confidence scores
✓ Image detection summaries
✓ Batch image processing
✓ Video detection
✓ H.264 MP4 output
✓ FastAPI integration
✓ Swagger documentation
```

The project is currently structured as a learning and portfolio project demonstrating how computer-vision inference can be integrated into a Python REST API.
