# CUDA Support Fix for Ladbon AI Desktop

This document explains the steps taken to fix CUDA/GPU support in the packaged version of Ladbon AI Desktop.

## Problem Description

The packaged (PyInstaller) version of Ladbon AI Desktop was failing to initialize CUDA support with an access violation error:

```
OSError: exception: access violation reading 0x0000000000000000
```

This error occurred in the `llama_cpp.llama_backend_init(False)` call, despite the correct DLLs being present in the packaged app.

## Solution

We've implemented a comprehensive fix with multiple approaches to ensure CUDA support works in the packaged application:

1. **CUDA Initialization Module** (`init_cuda.py`): A dedicated module that handles CUDA DLL loading and initialization before llama_cpp is imported.

2. **DLL Search Path Enhancement**: Adding all relevant directories to the DLL search path to ensure CUDA DLLs are found.

3. **Monkey Patching**: Patching the `llama_backend_init` function to handle different function signatures and provide better error handling.

4. **Multiple Launchers**: Different launcher scripts that set up the environment in various ways to ensure CUDA works.

5. **Diagnostic Tools**: Testing and diagnostic scripts to verify CUDA support.

## Using the Fixed Application

The following launchers are available in the packaged app directory:

1. **Launch with Advanced CUDA Support.bat**
   - Sets up all CUDA paths and runs an initialization script
   - Provides detailed logging of the CUDA setup process
   - Recommended for most users

2. **Launch with Direct CUDA Support.bat**
   - A simpler approach that just adds CUDA DLLs to the PATH
   - Use if the advanced launcher doesn't work

3. **Run CUDA Test.bat**
   - Runs a diagnostic script to check CUDA support
   - Creates a detailed log file with the results

## Technical Details

### The Root Cause

The main issue was that the PyInstaller packaging process affects how DLLs are loaded, particularly with complex dependencies like CUDA. When the application tries to initialize the CUDA backend via `llama_cpp.llama_backend_init(False)`, it was failing to find or properly initialize the required CUDA libraries.

### The Fix Components

#### 1. init_cuda.py

This module:
- Explicitly loads all CUDA DLLs before llama_cpp is imported
- Adds all relevant directories to the DLL search path
- Monkey patches the llama_backend_init function to handle different signatures and provide fallbacks

#### 2. cuda_hook.py

An initialization script that:
- Is run by the launcher before starting the main app
- Sets up the environment for CUDA support
- Loads init_cuda.py and initializes CUDA

#### 3. Advanced Launchers

The launchers perform these tasks:
- Add CUDA directories to the PATH environment variable
- Run initialization scripts
- Start the application with the proper environment

### Diagnostic Tools

#### test_cuda_in_packaged_app.py

This script:
- Lists all DLLs in the packaged app
- Tests CUDA initialization
- Tries to load a small model with and without CUDA
- Creates a detailed log file with results

## Troubleshooting

If CUDA support still doesn't work:

1. Check the log files in the `logs` directory
2. Ensure your NVIDIA drivers are up-to-date
3. Verify that CUDA 12.x is installed on your system
4. Try the different launcher scripts
5. Run the CUDA test script to get detailed diagnostics

## References

- llama-cpp-python documentation: https://github.com/abetlen/llama-cpp-python
- CUDA installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
- PyInstaller DLL handling: https://pyinstaller.org/en/stable/operating-mode.html#how-one-file-mode-works
