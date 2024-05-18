import cv2
import numpy as np

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

    # Set a fixed size for the output image
    maxWidth = 450
    maxHeight = 628

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

def enhance_image(image):
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control (0-100)

    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_contour_area = 0
area_threshold = 500  # Minimum change in area to reprocess the contour

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold where black is the foreground and white is the background
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours from the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = False  # Flag to indicate if an object has been found
    for contour in contours:
        current_area = cv2.contourArea(contour)
        if current_area > 1000 and abs(current_area - last_contour_area) > area_threshold:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the approximated contour has four points
            if len(approx) == 4:
                warped = four_point_transform(frame, approx.reshape(4, 2))
                sharpened_image = sharpen_image(warped)
                enhanced_image = enhance_image(sharpened_image)
                rotated_image = cv2.rotate(enhanced_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imshow("Sharpened Flattened Object", rotated_image)
                found = True
                last_contour_area = current_area  # Update last known area

    # Display the original and contours frame
    cv2.imshow("Original", frame)
    cv2.imshow("Contours", thresh)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('g'):
        resized_image = cv2.resize(rotated_image, (450, 628))  # Resize to 450x628
        cv2.imwrite('saved_image.png', resized_image)  # Save the image
        print("Image saved!")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
