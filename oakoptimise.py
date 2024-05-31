import depthai as dai
import cv2
import numpy as np
import time

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def improve_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Create a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(320, 240)  # Reduced frame size to increase FPS
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setFps(30)  # Increase the FPS setting of the camera
cam_rgb.initialControl.setManualFocus(130)  # Adjust this value to improve focus
cam_rgb.initialControl.setAutoExposureCompensation(5)  # Adjust exposure as needed

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Connect to the device and start the pipeline
with dai.Device(pipeline) as device:
    # Output queue will be used to get the rgb frames from the output defined above
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    while True:
        in_rgb = q_rgb.get()  # Get the frameset
        frame = in_rgb.getCvFrame()  # Convert the frame to OpenCV format

        # Increment frame count
        frame_count += 1

        # Calculate and display FPS every second
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold where black is the foreground and white is the background
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours from the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around contours
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                epsilon = 0.05 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    warped = four_point_transform(frame, approx.reshape(4, 2))
                    sharpened_image = sharpen_image(warped)
                    improved_image = improve_contrast(sharpened_image)
                    final_image = reduce_noise(improved_image)
                    cv2.imshow("Sharpened Flattened Object", final_image)

        # Display the results
        cv2.imshow('Original', frame)
        cv2.imshow('Threshold', thresh)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('p') and 'final_image' in locals():
            resized_image = cv2.resize(final_image, (450, 628))  # Resize to 450x628
            cv2.imwrite('saved_image.png', resized_image)  # Save the image
            print("Image saved!")

cv2.destroyAllWindows()
