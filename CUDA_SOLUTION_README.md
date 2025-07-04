# CUDA Support for Ladbon AI Desktop (Packaged App)

This document provides step-by-step instructions to fix CUDA support in the packaged version of Ladbon AI Desktop.

## Problem Summary

The packaged version of Ladbon AI Desktop (created with PyInstaller) fails to initialize the CUDA backend when loading models, resulting in an access violation error:

```
OSError: exception: access violation reading 0x0000000000000000
```

This occurs during the `llama_backend_init(False)` call in the llama-cpp-python library. The CLI/venv version works correctly with CUDA, but the packaged app fails.

## Root Cause

The issue is that the PyInstaller-packaged app cannot find or properly load the required CUDA DLLs (cudart64_12.dll, cublas64_12.dll, etc.) during runtime, even though they may be present on the system.

## Solution Steps

### 1. Run the Administrator-Elevated Fix Script

The `fix_cuda_admin.bat` script will:
- Request administrator privileges (required to copy files from Program Files)
- Copy the necessary CUDA DLLs from your NVIDIA CUDA Toolkit to the packaged app
- Create optimized launcher scripts for CUDA support
- Create diagnostic tools to verify CUDA DLL loading

```bash
cd c:\Users\ladfr\source\localai\src
fix_cuda_admin.bat
```

### 2. Run the CUDA Test Script

After the fix script completes, navigate to the packaged app directory and run the CUDA test:

```bash
cd c:\Users\ladfr\source\localai\src\dist\Ladbon AI Desktop
Run_CUDA_Test.bat
```

Check the generated log file in the `logs` directory to see if CUDA DLLs are being properly loaded.

### 3. Launch with CUDA Support

Use the newly created launcher with CUDA support:

```bash
cd c:\Users\ladfr\source\localai\src\dist\Ladbon AI Desktop
Launch_with_CUDA_Admin.bat
```

### 4. Apply Additional Runtime Patches (if needed)

If the issue persists, run the Python patch script to modify the runtime CUDA handling:

```bash
cd c:\Users\ladfr\source\localai\src
python patch_cuda_for_app.py
```

Then try launching with the Python CUDA launcher:

```bash
cd c:\Users\ladfr\source\localai\src\dist\Ladbon AI Desktop
Launch_with_Python_CUDA.bat
```

### 5. Direct CUDA Support

For a more direct approach, use the Direct CUDA Support launcher:

```bash
cd c:\Users\ladfr\source\localai\src\dist\Ladbon AI Desktop
Launch_with_Direct_CUDA_Support.bat
```

## Troubleshooting

If you continue to experience issues:

1. Check the logs in the `logs` directory for errors
2. Verify that CUDA DLLs are present in `_internal\cuda_dlls`
3. Confirm that the PATH environment variable includes the CUDA DLL directories
4. Test the CLI version with the same model to confirm CUDA works outside the packaged app

## Required CUDA DLLs

The following DLLs are essential for CUDA support:

- `cudart64_12.dll` - CUDA Runtime
- `cublas64_12.dll` - CUDA Basic Linear Algebra Subroutines
- `cublasLt64_12.dll` - CUDA Basic Linear Algebra Subroutines (lightweight)
- `nvrtc64_120.dll` - CUDA Runtime Compiler (optional)
- `cudnn64_8.dll` - CUDA Deep Neural Network Library (optional)

## Manual Fix

If the automated scripts don't work, you can manually:

1. Copy the CUDA DLLs from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin` to `dist\Ladbon AI Desktop\_internal\cuda_dlls`
2. Set the PATH environment variable to include this directory before launching the app

```bash
set PATH=C:\Users\ladfr\source\localai\src\dist\Ladbon AI Desktop\_internal\cuda_dlls;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
"Ladbon AI Desktop.exe" --use-cuda --verbose
```

## Technical Details

The packaged app needs to:

1. Find and load the CUDA DLLs before initializing llama-cpp-python
2. Correctly call the llama_backend_init function with CUDA support
3. Handle any errors gracefully if CUDA initialization fails

The fix scripts modify the runtime environment and patch key Python files to ensure these steps work correctly.
