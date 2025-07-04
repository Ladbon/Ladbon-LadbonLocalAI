# GPU/CUDA Support in Packaged Application

This document explains how to ensure CUDA/GPU support works in the packaged application.

## Understanding the Problem

The main issue is that when PyInstaller packages the application, the CUDA DLLs and the proper search paths for those DLLs are not always correctly set up, leading to failures in the GPU initialization and model loading.

## Solution

We've implemented several fixes to ensure CUDA/GPU support works properly:

1. **Enhanced Hook Script**: `hook-llama_cpp_enhanced.py` ensures all necessary DLLs and Python files from llama-cpp-python are correctly included in the package.

2. **Runtime CUDA Path Hook**: `cuda_path_hook.py` runs early during application startup to set up the correct DLL search paths.

3. **Diagnostic Tools**: 
   - `cuda_diagnostics.py` provides detailed diagnostics about CUDA installation and DLL loading
   - `test_llama_cpp_cuda.py` tests if llama-cpp-python is properly loaded with CUDA support

4. **Launcher Scripts**:
   - `setup_cuda_for_app.bat` sets up the environment for CUDA support
   - `test_cuda_integration.bat` tests both the development and packaged versions

## How to Use

### Building with GPU Support

1. Run the normal packaging process:
   ```
   python package.py
   ```

2. Set up CUDA support for the packaged app:
   ```
   setup_cuda_for_app.bat
   ```

3. Run the packaged app with CUDA support:
   ```
   dist\Ladbon AI Desktop\Launch_with_CUDA.bat
   ```

### Testing CUDA Support

Run the test script to verify CUDA is working in both environments:
```
test_cuda_integration.bat
```

## Troubleshooting

If GPU support is still not working:

1. Run the diagnostics script to get detailed information:
   ```
   dist\Ladbon AI Desktop\Check_CUDA.bat
   ```

2. Check the logs in `dist\Ladbon AI Desktop\logs\cuda_diagnostics.log` and `cuda_setup.log`.

3. Make sure your system has the right CUDA version installed.

4. Try copying the CUDA DLLs manually from your CUDA installation to `dist\Ladbon AI Desktop\_internal\cuda_dlls`.

## Requirements

- NVIDIA GPU with up-to-date drivers
- CUDA Toolkit installed (version 11.8 or higher recommended)
- llama-cpp-python built with CUDA support

## Important Notes

- The application first tries to use GPU if available, then falls back to CPU if necessary
- Some models may require specific CUDA versions - check compatibility
- Always check the logs if you encounter issues
