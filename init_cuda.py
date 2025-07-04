"""
CUDA initialization module for llama-cpp-python in packaged apps.

This module is designed to be included in the packaged app to ensure
proper CUDA initialization before llama_cpp is imported.
"""

import os
import sys
import glob
import logging
import ctypes
from ctypes import WinDLL
import time

# Set up logging
logger = logging.getLogger("init_cuda")

def ensure_cuda_dlls_loaded():
    """
    Explicitly load all CUDA DLLs before llama_cpp is imported
    to ensure they are properly initialized.
    
    This is necessary because the packaging process might 
    interfere with the normal DLL search path.
    """
    logger.info("Initializing CUDA DLLs...")
    
    # Dictionary of loaded DLLs to prevent double loading
    loaded_dlls = {}
    
    # Current directory - where the app is running from
    app_dir = os.path.abspath(os.path.dirname(sys.executable))
    
    # If running from PyInstaller bundle, use _MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        logger.info(f"Running from PyInstaller bundle: {meipass}")
        app_dir = meipass
    
    # Paths to search for CUDA DLLs
    search_paths = [
        # Packaged app directories
        app_dir,
        os.path.join(app_dir, "cuda_dlls"),
        os.path.join(app_dir, "_internal", "cuda_dlls"),
        os.path.join(app_dir, "_internal", "llama_cpp", "lib"),
        os.path.join(app_dir, "llama_cpp", "lib"),
        
        # System CUDA paths
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    ]
    
    # Add CUDA_PATH from environment if available
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        logger.info(f"Found CUDA_PATH in environment: {cuda_path}")
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.exists(cuda_bin):
            search_paths.insert(0, cuda_bin)
    
    # Add paths from PATH environment variable
    path_env = os.environ.get("PATH", "")
    path_dirs = path_env.split(os.pathsep)
    for path in path_dirs:
        if "cuda" in path.lower() and os.path.exists(path):
            if path not in search_paths:
                search_paths.append(path)
    
    # Add all search paths to the DLL search path
    for path in search_paths:
        if os.path.exists(path):
            try:
                # Use AddDllDirectory if available (Python 3.8+)
                if hasattr(os, 'add_dll_directory'):
                    dll_dir = os.add_dll_directory(path)
                    logger.info(f"Added DLL directory: {path}")
                # Otherwise, add to PATH
                elif path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
                    logger.info(f"Added to PATH: {path}")
            except Exception as e:
                logger.warning(f"Failed to add DLL directory {path}: {e}")
    
    # CUDA DLLs to load, in order of dependency
    cuda_dlls = [
        # Core CUDA runtime
        "cudart64_12.dll", "cudart64_120.dll", "cudart64_121.dll", 
        "cudart64_122.dll", "cudart64_123.dll", "cudart64_124.dll", 
        "cudart64_125.dll", "cudart64_126.dll",
        
        # CUDA libraries
        "cublas64_12.dll", "cublasLt64_12.dll", 
        "curand64_12.dll", "cusparse64_12.dll",
        
        # Other CUDA dependencies
        "nvrtc64_120_0.dll", "nvrtc64_121_0.dll", "nvrtc64_122_0.dll",
        "nvrtc64_123_0.dll", "nvrtc64_124_0.dll", "nvrtc64_125_0.dll",
        "nvrtc-builtins64_120.dll", "nvrtc-builtins64_121.dll",
        "nvrtc-builtins64_122.dll", "nvrtc-builtins64_123.dll",
        "nvrtc-builtins64_124.dll", "nvrtc-builtins64_125.dll",
    ]
    
    # Now search for and load all DLLs
    for dll_name in cuda_dlls:
        if dll_name in loaded_dlls:
            continue
        
        # Try to find the DLL
        dll_found = False
        for path in search_paths:
            if not os.path.exists(path):
                continue
                
            dll_path = os.path.join(path, dll_name)
            if os.path.exists(dll_path):
                try:
                    # Attempt to load the DLL
                    dll_handle = WinDLL(dll_path)
                    logger.info(f"Successfully loaded {dll_name} from {path}")
                    loaded_dlls[dll_name] = dll_path
                    dll_found = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {dll_path}: {e}")
        
        if not dll_found:
            logger.warning(f"Could not find or load {dll_name} in any search path")
    
    # Report on loading status
    if loaded_dlls:
        logger.info(f"Successfully loaded {len(loaded_dlls)} CUDA DLLs")
        return True
    else:
        logger.warning("No CUDA DLLs could be loaded")
        return False

def monkey_patch_llama_cpp():
    """
    Apply monkey patches to llama_cpp to ensure it can initialize
    with the CUDA backend properly.
    """
    try:
        logger.info("Applying monkey patches to llama_cpp...")
        
        # Try to import llama_cpp
        import llama_cpp
        
        # Store the original backend_init function
        original_backend_init = llama_cpp.llama_backend_init
        
        # Define our patched backend_init function that can handle different signatures
        def patched_backend_init(*args, **kwargs):
            numa = False
            if args:
                numa = args[0]
            elif 'numa' in kwargs:
                numa = kwargs['numa']
                
            logger.info(f"Patched llama_backend_init called with numa={numa}")
            
            # Try different calling patterns
            methods = [
                # Method 1: Call with args as-is
                lambda: original_backend_init(*args, **kwargs),
                # Method 2: Call with no args
                lambda: original_backend_init(),
                # Method 3: Try with opposite numa value if boolean was provided
                lambda: original_backend_init(not numa) if isinstance(numa, bool) else original_backend_init(),
                # Method 4: Force CPU mode and try original call pattern
                lambda: (setattr(os.environ, "LLAMA_CPP_CPU_ONLY", "1"), original_backend_init(*args, **kwargs))[1],
                # Method 5: Force CPU mode and try with no args
                lambda: (setattr(os.environ, "LLAMA_CPP_CPU_ONLY", "1"), original_backend_init())[1],
            ]
            
            # Try each method until one works
            last_error = None
            for i, method in enumerate(methods):
                try:
                    logger.info(f"Trying llama_backend_init method {i+1}")
                    result = method()
                    logger.info(f"llama_backend_init method {i+1} succeeded")
                    return result
                except Exception as e:
                    logger.warning(f"llama_backend_init method {i+1} failed: {e}")
                    last_error = e
            # If we get here, all methods failed
            logger.error("All llama_backend_init methods failed")
            if last_error:
                raise last_error
        
        # Apply the monkey patch
        llama_cpp.llama_backend_init = patched_backend_init
        logger.info("Successfully monkey-patched llama_backend_init")
        
        return True
    except ImportError as ie:
        logger.warning(f"Could not import llama_cpp: {ie}")
        return False
    except Exception as e:
        logger.error(f"Failed to monkey-patch llama_cpp: {e}")
        return False

def initialize_cuda():
    """
    Initialize CUDA environment for llama-cpp-python.
    
    Returns:
        bool: True if initialization was successful
    """
    try:
        # First, make sure we load all CUDA DLLs
        cuda_loaded = ensure_cuda_dlls_loaded()
        
        # Then try to monkey-patch llama_cpp
        patched = monkey_patch_llama_cpp()
        
        return cuda_loaded and patched
    except Exception as e:
        logger.error(f"CUDA initialization failed: {e}")
        return False
