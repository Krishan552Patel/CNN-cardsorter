import os
import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


def resilient_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except IOError as e:
        print(f"Could not load image {path}: {e}")
        return None

# Define a custom dataset class that uses the resilient loader
class ResilientImageFolder(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.class_to_idx = {}
        self.samples = []
        for class_name in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = len(self.class_to_idx)
                for img_filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_filename)
                    if resilient_pil_loader(img_path):
                        self.samples.append((img_path, self.class_to_idx[class_name]))

    def __getitem__(self, index):
        img_path, class_index = self.samples[index]
        image = resilient_pil_loader(img_path)
        if image is None:
            return torch.zeros(3, 224, 224), class_index  # Return a zero tensor if image failed to load
        if self.transform:
            image = self.transform(image)
        return image, class_index

    def __len__(self):
        return len(self.samples)

# Define transformations
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ResilientImageFolder(root='E:/CARDSpics/CARDdb/ENGLISH/Full_set', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN architecture
class CardCNN(nn.Module):
    def __init__(self):
        super(CardCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, len(dataset.class_to_idx))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = CardCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=25):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            if images is None:  
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100 * correct / total

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:  # Skip bad images
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / total
        val_accuracy = 100 * correct / total

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10)

# Save the model
torch.save(model.state_dict(), 'card_cnn.pth')
print("Model saved successfully.")
