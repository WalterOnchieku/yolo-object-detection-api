from pathlib import Path
import subprocess

import cv2
from ultralytics import YOLO



# Load the YOLO model once when this module is imported.
model = YOLO("yolov8n.pt")


def process_video(
    input_path: str,
    output_path: str,
    confidence_threshold: float = 0.4
):
    

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create the output directory if it doesn't exist.

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # OpenCV writes the processed frames here first.
    # FFmpeg will later convert this into the final MP4.

    temporary_video = (
        output_path.parent /
        f"{output_path.stem}_temp.avi"
    )

    # Open the input video.

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {input_path}"
        )

    # Read video information.

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Make sure the FPS is valid.
    if fps <= 0:
        cap.release()

        raise ValueError(
            "Could not determine video FPS."
        )

    # Create the temporary video writer.
    # MJPG is used because it is reliable with OpenCV.
    

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

    # Process each frame.
    frame_number = 0

    try:

        while True:

            # Read one frame from the video.
            ret, frame = cap.read()

            # Stop when there are no more frames.
            if not ret:
                break

            frame_number += 1

            # Run YOLO inference on the current frame.
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                verbose=False
            )

            result = results[0]

            # Process every detection found in the frame.
            for box in result.boxes:

                # Get the numerical class ID.
                class_id = int(
                    box.cls[0]
                )

                # Convert the class ID into a readable name.
                class_name = result.names[
                    class_id
                ]

                # Get the confidence score.
                confidence = float(
                    box.conf[0]
                )

                # Get bounding-box coordinates.
                x1, y1, x2, y2 = [
                    int(value)
                    for value in box.xyxy[0]
                ]

                # Draw the bounding box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 200, 100),
                    2
                )

                # Create the detection label.
                # Example: person 94%, car 87%
                label = (
                    f"{class_name} "
                    f"{confidence:.0%}"
                )

                # Draw the label above the bounding box.
                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(25, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 100),
                    2,
                    cv2.LINE_AA
                )

           
            # Write the processed frame to the temporary video.
            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    
    # Convert the temporary AVI into a final H.264 MP4.
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(temporary_video),
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path)
    ]

    try:

        subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    except FileNotFoundError:

        raise RuntimeError(
            "FFmpeg is not installed or "
            "cannot be found in PATH."
        )

    except subprocess.CalledProcessError as error:

        raise RuntimeError(
            f"FFmpeg failed to encode the video:\n"
            f"{error.stderr}"
        )

    finally:

        # Remove the temporary AVI file.
        temporary_video.unlink(
            missing_ok=True
        )

    # Return useful information to the caller.
    # FastAPI will eventually use this information
    # in its response.

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "processed_frames": frame_number
    }
