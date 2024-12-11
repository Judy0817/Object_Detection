from ultralytics import YOLO
import cv2
import os
import json
from sort import Sort  # Import SORT tracker

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use a YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt)

# Open the video file
video_path = 'videos/istockphoto-1336889543-640_adpp_is.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create output directory for frames with detections
output_folder = 'extract_frames'
os.makedirs(output_folder, exist_ok=True)

# Define the codec and create VideoWriter object for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out_video_path = 'output_video.mp4'
out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Create JSON file to store frame data
json_output_path = 'frame_data.json'
frame_data = []

# Initialize SORT tracker
tracker = Sort()

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error.")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Extract detected boxes and confidences
    detections = []
    for result in results[0].boxes:
        # Only keep detections with a confidence above a threshold (e.g., 0.5)
        if result.conf >= 0.5:
            box = result.xyxy.tolist()  # Bounding box coordinates [x1, y1, x2, y2]
            detections.append([box[0], box[1], box[2], box[3], result.conf])

    # Update tracker with the current detections
    trackers = tracker.update(detections)

    # Annotate the frame with detection results and tracking IDs
    for track in trackers:
        x1, y1, x2, y2, track_id = track
        class_name = model.names[int(results[0].boxes[track[4]].cls)]  # Get class name by index
        confidence = track[4]  # Confidence score
        # Draw the bounding box and label with the assigned ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)} {class_name} ({confidence:.2f})',
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the annotated frame to the output folder
    output_frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(output_frame_path, frame)

    # Write the annotated frame to the output video
    out.write(frame)

    # Extract detection details for the frame with tracking IDs
    detections = []
    for track in trackers:
        detections.append({
            'id': int(track[4]),  # Tracker ID
            'class': model.names[int(results[0].boxes[track[4]].cls)],
            'confidence': float(track[3]),  # Confidence
            'box': [track[0], track[1], track[2], track[3]]  # Bounding box
        })

    # Append frame data to JSON structure
    frame_data.append({
        'frame': frame_count,
        'detections': detections
    })

    frame_count += 1

    # Optional: Display the frame (comment out if not needed)
    cv2.imshow('YOLOv8 Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write the frame data to the JSON file
with open(json_output_path, 'w') as json_file:
    json.dump(frame_data, json_file, indent=4)

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames. Annotated frames saved in '{output_folder}', video saved as '{out_video_path}', and frame data saved in '{json_output_path}'")
