import torch

print(f"✅ CUDA Available: {torch.cuda.is_available()}")
print(f"🔹 CUDA Device Count: {torch.cuda.device_count()}")
print(f"🔹 CUDA Device Name: {torch.cuda.get_device_name(0)}")
print(f"🔹 Current Device: {torch.cuda.current_device()}")
print(torch.version.cuda)
print(torch.__version__)