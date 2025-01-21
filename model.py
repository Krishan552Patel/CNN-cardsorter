
import os
import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import shutil
import threading

# ==================== ✅ GPU CHECK ====================
print("🔍 Checking CUDA/GPU availability...")

print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"🔹 CUDA Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"🔹 CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"🔹 Current Device: {torch.cuda.current_device()}")
print(f"🔹 Installed CUDA Version (PyTorch): {torch.version.cuda}")
print(f"🔹 PyTorch Version: {torch.__version__}")

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Using device: {device}")

# ✅ Enable cuDNN Optimizations for Faster Training
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ==================== SETUP DIRECTORY ====================
FABCARD =  FABCARD = "/mnt/dataset/Processed_Images"                       #forlocal machine #"E:/card/Processed_Images"
checkpoint_dir = "E:/checkpoints"
# 🔥 Check if required directories exist
if not os.path.exists("E:/"):
    print("⚠️ Warning: E:/ drive does not exist. Switching to C:/")
    FABCARD = "C:/card/Processed_Images"
    checkpoint_dir = "C:/checkpoints"

# 🔥 Ensure dataset directory exists
if not os.path.exists(FABCARD):
    print(f"❌ Dataset directory {FABCARD} not found. Creating it now...")
    os.makedirs(FABCARD, exist_ok=True)
else:
    print(f"✅ Dataset directory exists: {FABCARD}")

# 🔥 Ensure checkpoint directory exists
if not os.path.exists(checkpoint_dir):
    print(f"❌ Checkpoint directory {checkpoint_dir} not found. Creating it now...")
    os.makedirs(checkpoint_dir, exist_ok=True)
else:
    print(f"✅ Checkpoint directory exists: {checkpoint_dir}")

# ==================== CUSTOM DATASET CLASS ====================
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
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

# ==================== DEFINE IMAGE TRANSFORMATIONS ====================
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== FUNCTION TO LOAD CHECKPOINT SAFELY ====================
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        if os.path.getsize(checkpoint_path) > 0:
            try:
                print("🔄 Resuming from latest checkpoint...")
                checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint.get("epoch", 0)
                start_batch = checkpoint.get("batch", 0)
                print(f"✅ Resumed from Epoch {start_epoch+1}, Batch {start_batch+1}")
                return start_epoch, start_batch
            except Exception as e:
                print(f"❌ Failed to load checkpoint ({checkpoint_path}): {e}")
                print("🚨 Corrupted checkpoint detected. Deleting and starting from scratch.")
                os.remove(checkpoint_path)  # Delete corrupted checkpoint
        else:
            print("❌ Checkpoint file is empty. Starting from scratch.")
            os.remove(checkpoint_path)  # Delete empty file
    else:
        print("❌ No valid checkpoint found. Starting from scratch.")
    return 0, 0  # Start from scratch
def save_checkpoint(epoch, batch_idx, model, optimizer, checkpoint_path, temp_checkpoint_path):
    """Asynchronously saves a checkpoint without blocking training."""
    torch.save({
        "epoch": epoch,
        "batch": batch_idx + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, temp_checkpoint_path)
    shutil.move(temp_checkpoint_path, checkpoint_path)
    print(f"💾 Updated latest checkpoint: {checkpoint_path}")

# ==================== DEFINE CNN ARCHITECTURE ====================
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

# ==================== TRAINING FUNCTION WITH CHECKPOINTING ====================
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10, batch_checkpoint_interval=500):
    checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
    temp_checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, "latest_checkpoint_tmp.pth"))

    # 🔄 Load checkpoint safely
    start_epoch, start_batch = load_checkpoint(model, optimizer, checkpoint_path)

    # ✅ Move model to GPU
    model.to(device)

    # ✅ Initialize AMP GradScaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        print(f"\n🚀 Epoch {epoch + 1}/{epochs} Started...")  

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx < start_batch:
                print(f"🔄 Skipping already processed Batch {batch_idx + 1}")
                continue  

            print(f"🟢 Processing Batch {batch_idx + 1}/{len(train_loader)} (Epoch {epoch + 1}) ...")

            # ✅ Move images & labels to GPU
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # ✅ Enable Mixed Precision Training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # ✅ Scale loss and perform backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f"✅ Batch {batch_idx + 1}/{len(train_loader)} completed | Loss: {loss.item():.4f}")

            # 🔹 Save checkpoint only every batch_checkpoint_interval batches
            if (batch_idx + 1) % batch_checkpoint_interval == 0:
                print(f"💾 Saving checkpoint at Batch {batch_idx + 1}...")
                threading.Thread(target=save_checkpoint, args=(epoch, batch_idx, model, optimizer, checkpoint_path, temp_checkpoint_path)).start()

    # 🔹 Save final checkpoint after each epoch
    print(f"💾 Saving final checkpoint for Epoch {epoch + 1}...")
    save_checkpoint(epoch, batch_idx, model, optimizer, checkpoint_path, temp_checkpoint_path)

    print(f"✅ Training complete. Final checkpoint saved at: {checkpoint_path}")

# ==================== MAIN EXECUTION WRAPPER ====================
if __name__ == "__main__":
    print("📂 Loading dataset...")
    
    dataset = ResilientImageFolder(root=FABCARD, transform=transform)

    print(f"✅ Dataset loaded! Found {len(dataset)} images.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("📂 Creating data loaders...")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)




    print("✅ Data loaders ready!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    print("🔄 Initializing model...")
    
    num_classes = len(dataset.class_to_idx)
    model = CardCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"✅ Model initialized with {num_classes} classes.")

    print("🚀 Starting training...")

    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10, batch_checkpoint_interval=500)


    print("✅ Training completed!")