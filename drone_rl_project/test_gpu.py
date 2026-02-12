import torch

# Add this to your train_simple.py to verify GPU is being used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")