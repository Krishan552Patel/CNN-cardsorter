import os 
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder

# Define the CNN Architecture
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
def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = dataset.classes[predicted[0]]
    end_time = time.time()
    
    elapsed_time = round(end_time - start_time,4)
    return predicted_class, elapsed_time
    
# Example usage
test_folder = 'E:/testphotodata/test'
for image_file in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_file)
    if image_path.endswith(".png") or image_path.endswith(".jpg"):
        predicted_class, elapsed_time = predict_image(image_path, model, device)
        print(f'Image: {image_file} -> Predicted Class: {predicted_class} (Time: {elapsed_time} sec)')