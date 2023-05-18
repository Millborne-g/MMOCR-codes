import cv2

def frame_skip(video_path, output_path, skip):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create a video writer object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_counter = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        if not ret:
            break

        frame_counter += 1

        # Skip frames
        if frame_counter % skip != 0:
            continue

        # Write the frame to the output video
        out.write(frame)

    # Release the video capture and writer objects
    video.release()
    out.release()

    print("Frame skipping completed.")

# Provide the path to the input video file
input_video_path = "Fasten30mins6.mp4"

# Provide the path to the output video file
output_video_path = "Fasten30mins6_FrameSkip.mp4"

# Define the number of frames to skip
frame_skip_value = 5

# Call the frame_skip function
frame_skip(input_video_path, output_video_path, frame_skip_value)
