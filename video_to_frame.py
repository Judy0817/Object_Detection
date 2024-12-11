import cv2
import os

def extract_frames(video_path, output_folder, interval_seconds=1):
    """
    Extract frames from a video at configurable intervals.

    :param video_path: Path to the video file.
    :param output_folder: Directory to save extracted frames.
    :param interval_seconds: Interval between extracted frames in seconds.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return

    # Get video information
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(fps * interval_seconds)

    # Extract frames at specified intervals
    frame_count = 0
    frame_number = 0

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break  # End of video

        if frame_count % interval_frames == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_filename, frame)  # Save frame as JPG
            print(f"Saved {frame_filename}")
            frame_number += 1

        frame_count += 1

    video_capture.release()
    print("Frame extraction complete.")

# Example Usage
video_path = r'D:\USJ\Semester 7\FYP\Data Lake\videos\istockphoto-1336889543-640_adpp_is.mp4'
output_folder = r'D:\USJ\Semester 7\FYP\Data Lake\extracted_frames'  # Folder where frames will be saved
interval_seconds = 0.5  # Extract one frame every 2 seconds

extract_frames(video_path, output_folder, interval_seconds)
