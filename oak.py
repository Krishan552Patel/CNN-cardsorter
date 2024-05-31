import depthai as dai
import cv2
import time

# Create a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the rgb frames from the output defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    frame_count = 0
    start_time = time.time()

    while True:
        in_rgb = q_rgb.get()  # Get the frameset
        frame = in_rgb.getCvFrame()  # Convert the frame to OpenCV format
        frame_count += 1

        # Calculate and display FPS every second
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()
        
        # Draw a box in the center of the frame
        height, width, _ = frame.shape
        top_left = (width // 2 - 200, height // 2 -265)
        bottom_right = (width // 2 + 50, height // 2 + 280)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        
        # Apply Canny edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Display the original frame and the edges
        cv2.imshow("RGB Frame", frame)
        cv2.imshow("Edges", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
