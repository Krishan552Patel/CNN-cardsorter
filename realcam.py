import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

# Function to order points for perspective transform
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left point
    rect[2] = pts[np.argmax(s)]  # bottom-right point
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right point
    rect[3] = pts[np.argmax(diff)]  # bottom-left point
    return rect

# Function to perform the four point transformq
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxWidth = 450  # fixed width to 450
    maxHeight = 625  # fixed height to 625

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return warped

# Function to sharpen images
def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# CNN architecture definition
class CardCNN(nn.Module):
    def __init__(self, num_classes):
        super(CardCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset classes for label mapping
dataset = ImageFolder(root='E:/testphotodata/outsider_card_list')
num_classes = len(dataset.classes)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = CardCNN(num_classes).to(device)
model.load_state_dict(torch.load('E:/CNNmodel/card_cnn.pth'))
model.eval()  # Set the model to evaluation mode for prediction

# Prediction function
def predict_image(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(image)  # Convert OpenCV image to PIL image
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = dataset.classes[predicted[0]]

    return predicted_class

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

last_contour_area = 0
area_threshold = 500  # Change in area to process the contour

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
    found = False

    for contour in contours:
        current_area = cv2.contourArea(contour)
        if current_area > 1000 and abs(current_area - last_contour_area) > area_threshold:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the approximated contour has four points
            if len(approx) == 4:
                warped = four_point_transform(frame, approx.reshape(4, 2))
                sharpened_image = sharpen_image(warped)
                rotated_image = cv2.rotate(sharpened_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                predicted_class = predict_image(rotated_image, model, device)
                cv2.imshow("Sharpened Flattened Object", rotated_image)
                resized_image = cv2.resize(rotated_image, (450, 628))  # Resize to 450x628
                

                print(f'Predicted Class: {predicted_class}')
                found = True
                last_contour_area = current_area  # update last known area

    # Display the original frame
    cv2.imshow("Original", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
 

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()