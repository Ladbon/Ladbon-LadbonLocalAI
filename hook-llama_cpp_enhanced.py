"""
Hook script for PyInstaller to ensure proper llama_cpp DLL handling.
This script is executed during PyInstaller build and helps include all necessary DLLs.
"""

import os
import sys
import glob
from pathlib import Path
import PyInstaller.__main__

print("Running llama_cpp hook for PyInstaller...")

datas = []

def add_data_files(src_dir, dst_dir):
    """Add data files from src_dir to dst_dir in the package"""
    if not os.path.exists(src_dir):
        print(f"Source directory not found: {src_dir}")
        return []
    
    files = []
    for file in glob.glob(os.path.join(src_dir, "*")):
        if os.path.isfile(file):
            print(f"Adding file to package: {file} -> {dst_dir}")
            files.append((file, dst_dir))
    
    return files

# Try to find llama_cpp in the environment
try:
    import llama_cpp
    llama_cpp_dir = os.path.dirname(llama_cpp.__file__)
    print(f"Found llama_cpp module at: {llama_cpp_dir}")
    
    # Get the lib directory
    lib_dir = os.path.join(llama_cpp_dir, "lib")
    if os.path.exists(lib_dir):
        print(f"Found llama_cpp lib directory at: {lib_dir}")
        
        # Add all files from the lib directory to the package
        datas.extend(add_data_files(lib_dir, os.path.join("llama_cpp", "lib")))
    else:
        print("Warning: llama_cpp lib directory not found")
    
    # Add the dist-info directory
    llama_cpp_parent = os.path.dirname(llama_cpp_dir)
    dist_info_dirs = list(Path(llama_cpp_parent).glob("llama_cpp_python*.dist-info"))
    
    if dist_info_dirs:
        dist_info_dir = str(dist_info_dirs[0])
        print(f"Found dist-info directory: {dist_info_dir}")
        
        # Add all files from the dist-info directory to the package
        dist_info_name = os.path.basename(dist_info_dir)
        datas.extend(add_data_files(dist_info_dir, dist_info_name))
    else:
        print("Warning: llama_cpp_python dist-info directory not found")
    
    # Copy all Python files from llama_cpp
    for py_file in glob.glob(os.path.join(llama_cpp_dir, "*.py")):
        datas.append((py_file, "llama_cpp"))
    
    # Look for subdirectories in llama_cpp and include them
    for item in os.listdir(llama_cpp_dir):
        item_path = os.path.join(llama_cpp_dir, item)
        if os.path.isdir(item_path) and item != "__pycache__" and item != "lib":
            # Add all files from subdirectories
            for root, _, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(os.path.dirname(file_path), llama_cpp_dir)
                    target_dir = os.path.join("llama_cpp", rel_path)
                    datas.append((file_path, target_dir))
    
    # Look for CUDA DLLs in various locations
    cuda_dlls = []
    
    # Check CUDA paths in environment
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path and os.path.exists(os.path.join(cuda_path, "bin")):
        cuda_bin = os.path.join(cuda_path, "bin")
        print(f"Found CUDA bin directory in environment: {cuda_bin}")
        
        # Look for critical CUDA DLLs
        for pattern in ["cudart64*.dll", "cublas64*.dll", "cublasLt64*.dll", "curand64*.dll"]:
            matches = glob.glob(os.path.join(cuda_bin, pattern))
            cuda_dlls.extend(matches)
    
    # Check common CUDA installation locations on Windows
    if sys.platform == "win32":
        base_dirs = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files\NVIDIA Corporation"
        ]
        
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                # Look for version subdirectories
                for item in os.listdir(base_dir):
                    version_dir = os.path.join(base_dir, item)
                    if os.path.isdir(version_dir) and ("v" in item or "." in item):
                        bin_dir = os.path.join(version_dir, "bin")
                        if os.path.exists(bin_dir):
                            print(f"Found CUDA installation: {bin_dir}")
                            
                            # Look for critical CUDA DLLs
                            for pattern in ["cudart64*.dll", "cublas64*.dll", "cublasLt64*.dll", "curand64*.dll"]:
                                matches = glob.glob(os.path.join(bin_dir, pattern))
                                cuda_dlls.extend(matches)
    
    # Add found CUDA DLLs to the package
    if cuda_dlls:
        print(f"Found {len(cuda_dlls)} CUDA DLLs to include")
        for dll in cuda_dlls:
            print(f"  - Adding CUDA DLL: {dll}")
            datas.append((dll, "cuda_dlls"))
    else:
        print("No CUDA DLLs found to include")

except ImportError as e:
    print(f"Warning: Could not import llama_cpp: {e}")
    datas = []

print(f"Returning {len(datas)} data files for PyInstaller")
