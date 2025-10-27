import torch
import subprocess

def check_cuda_torch():
    print("🔍 Checking CUDA availability via PyTorch...")
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f" - CUDA device count  : {torch.cuda.device_count()}")
        print(f" - Current device     : {torch.cuda.current_device()}")
        print(f" - Device name        : {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("❌ CUDA is NOT available in this environment.")
        print(" - This may be due to: no GPU, missing drivers, or CUDA not visible to Python.")

def check_nvidia_smi():
    print("\n🔍 Checking system-level GPU status via nvidia-smi...")
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf-8")
        print("✅ NVIDIA-SMI output:\n")
        print(output)
    except FileNotFoundError:
        print("❌ nvidia-smi not found. This system may not have NVIDIA drivers installed.")
    except subprocess.CalledProcessError as e:
        print("❌ nvidia-smi failed to run. Error output:")
        print(e.output.decode("utf-8"))

if __name__ == "__main__":
    print("=== CUDA + GPU Availability Check ===\n")
    check_cuda_torch()
    check_nvidia_smi()