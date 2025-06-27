"""
PyInstaller runtime hook to patch NumPy CPU dispatcher before it can cause issues
and handle DLL loading in onefile mode
"""

import os
import sys
import types
import shutil
import ctypes
from datetime import datetime

# Import type stubs to silence IDE warnings
try:
    import pyinstaller_types  # type: ignore # noqa
except ImportError:
    pass  # Will be absent at runtime in bundle

# Create a debug log file
debug_log_path = os.path.join(os.getcwd(), f'numpy_hook_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
with open(debug_log_path, 'w') as f:
    f.write(f"NumPy hook running at {datetime.now()}\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Working directory: {os.getcwd()}\n")

# Set up environment variables to restrict threading
for var_name, value in {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", 
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1"
}.items():
    os.environ[var_name] = value
    with open(debug_log_path, 'a') as f:
        f.write(f"Set {var_name}={value}\n")

# Handle directory structure for llama_cpp in PyInstaller onefile mode
if hasattr(sys, "_MEIPASS"):
    # Get the PyInstaller bundle directory
    try:
        from pyinstaller_types import get_bundle_dir
        meipass_dir = get_bundle_dir()
    except ImportError:
        # Fall back if type stubs aren't available
        meipass_dir = sys._MEIPASS  # type: ignore
        
    with open(debug_log_path, 'a') as f:
        f.write(f"Running in PyInstaller bundle at: {meipass_dir}\n")
    
    # Create llama_cpp/lib directory structure
    try:
        llama_cpp_dir = os.path.join(meipass_dir, "llama_cpp")
        lib_dir = os.path.join(llama_cpp_dir, "lib")
        
        with open(debug_log_path, 'a') as f:
            f.write(f"Creating directories: {llama_cpp_dir} and {lib_dir}\n")
        
        os.makedirs(lib_dir, exist_ok=True)
        
        # Look for DLLs in the root directory and copy them to the lib directory
        if os.name == 'nt':
            dll_files = [f for f in os.listdir(meipass_dir) if f.lower().endswith('.dll')]
            with open(debug_log_path, 'a') as f:
                f.write(f"Found {len(dll_files)} DLLs in _MEIPASS directory\n")
            
            # Copy llama-cpp related DLLs to the lib directory
            for dll in dll_files:
                if any(name in dll.lower() for name in ["ggml", "llama", "llava"]):
                    src = os.path.join(meipass_dir, dll)
                    dst = os.path.join(lib_dir, dll)
                    with open(debug_log_path, 'a') as f:
                        f.write(f"Copying {dll} to {dst}\n")
                    shutil.copy2(src, dst)
        
            # Add the lib directory to the DLL search path
            try:
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                lib_dir_c = ctypes.c_wchar_p(lib_dir)
                handle = kernel32.AddDllDirectory(lib_dir_c)
                if handle:
                    with open(debug_log_path, 'a') as f:
                        f.write(f"Successfully added DLL directory: {lib_dir}\n")
                else:
                    error = ctypes.get_last_error()
                    with open(debug_log_path, 'a') as f:
                        f.write(f"Failed to add DLL directory, error code: {error}\n")
            except Exception as e:
                with open(debug_log_path, 'a') as f:
                    f.write(f"Error adding DLL directory: {e}\n")
        
        # Also add to PATH environment variable as a fallback
        os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')
        with open(debug_log_path, 'a') as f:
            f.write(f"Added {lib_dir} to PATH environment\n")
    
    except Exception as e:
        with open(debug_log_path, 'a') as f:
            f.write(f"Error handling directory structure: {e}\n")

# Create a dummy module to prevent NumPy from initializing its CPU dispatcher
if 'numpy' not in sys.modules and 'numpy.core._multiarray_umath' not in sys.modules:
    # Create a dummy module with the required structure
    dummy = types.ModuleType('numpy.core._multiarray_umath')
    
    # Define functions/values that might be accessed during NumPy initialization
    # These are dynamically added at runtime and not checked by type checkers
    setattr(dummy, '__cpu_features__', {})
    setattr(dummy, '__cpu_baseline__', [])
    setattr(dummy, '__cpu_dispatch__', [])
    
    def dummy_function(*args, **kwargs):
        return None
    
    # Define common functions that might be called
    setattr(dummy, 'get_cpu_features', dummy_function)
    setattr(dummy, 'implement_cpu_features', dummy_function)
    
    # Add the module to sys.modules to prevent real module from loading
    sys.modules['numpy.core._multiarray_umath'] = dummy
    
    with open(debug_log_path, 'a') as f:
        f.write("Created dummy numpy.core._multiarray_umath module\n")

with open(debug_log_path, 'a') as f:
    f.write("Runtime hook completed\n")
