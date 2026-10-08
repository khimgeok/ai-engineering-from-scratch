import torch
import time

size = 5000  # we multiply two 5000 x 5000 grids of random numbers

# ---------- CPU ----------
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu          # "@" means matrix multiplication
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

# ---------- GPU (Apple MPS or NVIDIA CUDA) ----------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = None

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

if device:
    a_gpu = a_cpu.to(device)   # copy the data into GPU memory
    b_gpu = b_cpu.to(device)

    _ = a_gpu @ b_gpu          # warm-up run (first GPU call has setup cost)
    sync()

    start = time.time()
    c_gpu = a_gpu @ b_gpu
    sync()                     # wait until the GPU has truly finished
    gpu_time = time.time() - start
    print(f"GPU ({device}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")
else:
    print("No GPU found. Try this on Google Colab.")
