import cv2
import numpy as np

# Open the default camera
cap = cv2.VideoCapture()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

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

    # Draw all contours
    result = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Original', frame)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Contours', result)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
