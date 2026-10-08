import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA (NVIDIA) available: {torch.cuda.is_available()}")
print(f"MPS (Apple GPU) available: {torch.backends.mps.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using: {device}")
