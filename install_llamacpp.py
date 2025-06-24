import os
import subprocess
import sys
import platform

print("Installing llama-cpp-python 0.3.9 with correct settings...")

# Determine if CUDA should be enabled
use_cuda = False
try:
    # Simple check for NVIDIA GPU
    if platform.system() == "Windows":
        output = subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.DEVNULL)
        use_cuda = True
        print("NVIDIA GPU detected, will configure for CUDA support")
except:
    print("No NVIDIA GPU detected or nvidia-smi not available, using CPU only")

# Set environment variables for the build
env = os.environ.copy()

if use_cuda:
    # Use the correct CUDA flag for 0.3.9
    env["CMAKE_ARGS"] = "-DGGML_CUDA=ON"
    print("Building with CUDA support (GGML_CUDA=ON)")
else:
    # Explicitly disable CUDA to avoid any issues
    env["CMAKE_ARGS"] = "-DGGML_CUDA=OFF"
    print("Building without CUDA support")

# Force cmake to rebuild
env["FORCE_CMAKE"] = "1"

# Run the pip install command
cmd = [
    sys.executable, "-m", "pip", "install", 
    "llama-cpp-python==0.3.9",
    "--no-cache-dir",
    "--verbose",
    "--force-reinstall"
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, env=env)

if result.returncode == 0:
    print("\nSuccessfully installed llama-cpp-python 0.3.9!")
else:
    print("\nInstallation failed with error code:", result.returncode)
    print("\nFalling back to CPU-only version 0.2.56...")
    
    # Fall back to the known working version
    fallback_cmd = [
        sys.executable, "-m", "pip", "install",
        "llama-cpp-python==0.2.56",
        "--no-cache-dir"
    ]
    subprocess.run(fallback_cmd)