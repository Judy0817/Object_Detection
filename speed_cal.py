import cv2
import numpy as np
import datetime
from ultralytics import YOLO
import supervision as sv
import os

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# LineZone and Annotators
LINE_START = sv.Point(150, 1000)
LINE_END = sv.Point(1100, 100)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

line_annotator = sv.LineZoneAnnotator(
    thickness=4, 
    text_thickness=4, 
    text_scale=2
)
box_annotator = sv.BoxAnnotator(
    thickness=2  # Adjust line thickness if available
)

# For object speed tracking
object_last_positions = {}
object_speeds = {}

# Constants for speed calculation
FRAME_RATE = 30  # Change according to your video frame rate
PIXEL_TO_REAL_SCALE = 0.05  # Pixels to meters (calibrate based on camera setup)

def calculate_speed(object_id, new_position):
    global object_last_positions
    speed = 0
    if object_id in object_last_positions:
        last_position, last_time = object_last_positions[object_id]
        distance = np.linalg.norm(np.array(new_position) - np.array(last_position))
        time_elapsed = 1 / FRAME_RATE
        speed = (distance * PIXEL_TO_REAL_SCALE) / time_elapsed  # Speed in meters/second
    object_last_positions[object_id] = (new_position, datetime.datetime.now())
    return speed

# Process video from CCTV
def process_video(video_path):
    global object_last_positions, object_speeds
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object tracking
        results = model.track(source=frame, tracker="bytetrack.yaml", persist=True, show=False)

        for result in results:
            detections = []
            if result.boxes.id is not None:
                for box, id, conf, class_id in zip(result.boxes.xyxy, result.boxes.id, result.boxes.conf, result.boxes.cls):
                    tracker_id = int(id)
                    x1, y1, x2, y2 = box.tolist()
                    position = ((x1 + x2) // 2, (y1 + y2) // 2)
                    speed = calculate_speed(tracker_id, position)

                    # Corrected way to create detection object
                    detection = sv.Detection(  # Corrected to use sv.Detection with a capital D
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=conf,
                        class_id=int(class_id),
                        tracker_id=tracker_id
                    )
                    detections.append(detection)

                    label = f"ID: {tracker_id}, {model.model.names[class_id]} {conf:.2f}, Speed: {speed:.2f} m/s"
                    labels = [label]

            # Convert to supervision Detections format
            detections = sv.Detections(detections)

            # Trigger line counter and annotate the frame
            line_counter.trigger(detections=detections)
            line_annotator.annotate(frame=frame, line_counter=line_counter)
            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels,
            )

        # Save annotated frame to output video
        out.write(frame)
        cv2.imshow("CCTV Footage", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    # Provide the path to your CCTV footage
    CCTV_VIDEO_PATH = "videos/istockphoto-1336889543-640_adpp_is.mp4"
    process_video(CCTV_VIDEO_PATH)