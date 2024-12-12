from ultralytics import YOLO
import cv2
import os
import json

# Load the YOLOv8 model
model = YOLO('model/yolov8n.pt')  # Use a YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt)

# Open the video file
video_path = 'inputs/test2.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames Per Second (FPS) of the video: {fps}")

# Create output directory for frames with detections
output_folder = 'extract_frames'
os.makedirs(output_folder, exist_ok=True)

# Define the codec and create VideoWriter object for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out_video_path = 'outputs/output_video_test2.mp4'
out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Create JSON file to store frame data
json_output_path = 'outputs/frame_data.json'
frame_data = []

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error.")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Save the annotated frame to the output folder
    output_frame_path = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(output_frame_path, annotated_frame)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Extract detection details for the frame
    detections = []
    for result in results[0].boxes:
        detections.append({
            'class': model.names[int(result.cls)],
            'confidence': float(result.conf),
            'box': result.xyxy.tolist()  # Bounding box coordinates
        })

    # Append frame data to JSON structure
    frame_data.append({
        'frame': frame_count,
        'detections': detections
    })

    frame_count += 1

    # Optional: Display the frame (comment out if not needed)
    cv2.imshow('YOLOv8 Detection', annotated_frame)

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