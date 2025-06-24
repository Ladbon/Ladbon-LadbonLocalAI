import os
import subprocess
import sys
import platform

print("Reinstalling llama-cpp-python...")

# --- Configuration ---
TARGET_LLAMA_CPP_PYTHON_VERSION = "0.3.9" # Or your preferred version that works with the patch
# --- End Configuration ---

def run_command(command):
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.check_call(command)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Command {command[0]} not found. Is it in your PATH?")
        return False

# 1. Uninstall current version
print("\n--- Uninstalling current llama-cpp-python ---")
run_command([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])

# 2. Attempt to install with CUDA support
print(f"\n--- Attempting to install llama-cpp-python version {TARGET_LLAMA_CPP_PYTHON_VERSION} with CUDA ---")

# Set environment variables for compilation
# For 0.2.x use -DLLAMA_CUBLAS=on, for 0.3.x+ use -DGGML_CUDA=ON
# Assuming TARGET_LLAMA_CPP_PYTHON_VERSION is 0.3.x or newer
cmake_args_cuda = "-DGGML_CUDA=ON" 
# You might need to specify CUDA architectures if auto-detection fails, e.g. -DCMAKE_CUDA_ARCHITECTURES=75;86
# For RTX A1000 Laptop GPU, architecture is Ampere (86).
# cmake_args_cuda += " -DCMAKE_CUDA_ARCHITECTURES=86" # Uncomment and adjust if needed

original_cmake_args = os.environ.get("CMAKE_ARGS")
original_force_cmake = os.environ.get("FORCE_CMAKE")

os.environ["CMAKE_ARGS"] = cmake_args_cuda
os.environ["FORCE_CMAKE"] = "1"

install_command_cuda = [
    sys.executable, "-m", "pip", "install",
    f"llama-cpp-python=={TARGET_LLAMA_CPP_PYTHON_VERSION}",
    "--no-cache-dir", "--force-reinstall", "--verbose"
]

if run_command(install_command_cuda):
    print("\nSuccessfully installed llama-cpp-python with CUDA support attempt.")
else:
    print("\nCUDA installation failed or was skipped. Falling back to CPU-only version...")
    
    # Reset environment variables
    if original_cmake_args is None:
        os.environ.pop("CMAKE_ARGS", None)
    else:
        os.environ["CMAKE_ARGS"] = original_cmake_args
    
    if original_force_cmake is None:
        os.environ.pop("FORCE_CMAKE", None)
    else:
        os.environ["FORCE_CMAKE"] = original_force_cmake

    print(f"\n--- Installing llama-cpp-python version {TARGET_LLAMA_CPP_PYTHON_VERSION} (CPU only) ---")
    install_command_cpu = [
        sys.executable, "-m", "pip", "install",
        f"llama-cpp-python=={TARGET_LLAMA_CPP_PYTHON_VERSION}",
        "--no-cache-dir", "--force-reinstall"
    ]
    if run_command(install_command_cpu):
        print("\nSuccessfully installed CPU-only version of llama-cpp-python.")
    else:
        print("\nERROR: Failed to install CPU-only version of llama-cpp-python.")
        print("Please check your Python environment and build tools (like Visual Studio C++ Build Tools).")

# Clean up environment variables set by this script
if original_cmake_args is None:
    os.environ.pop("CMAKE_ARGS", None)
else:
    os.environ["CMAKE_ARGS"] = original_cmake_args

if original_force_cmake is None:
    os.environ.pop("FORCE_CMAKE", None)
else:
    os.environ["FORCE_CMAKE"] = original_force_cmake

print("\nInstallation process complete. Please restart your application.")
print("The runtime patch in gui_app.py will attempt to handle backend initialization.")

if __name__ == "__main__":
    print("Installation process complete. Please restart your application.")

#
# if __name__ == "__main__":
#    main() # Assuming you might wrap this in a main function if you add more logic
