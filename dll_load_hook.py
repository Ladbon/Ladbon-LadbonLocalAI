import os
import sys
import ctypes
import logging
import glob
from pathlib import Path

# Set up basic logging to a file for debugging the hook itself
log_file_path = os.path.join(os.path.expanduser("~"), "ladbon_ai_hook_debug.log")
logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.info("Executing runtime hook: dll_load_hook.py")

def get_meipass_path():
    """Gets the PyInstaller _MEIPASS directory.
    
    Note: sys._MEIPASS is a special attribute added by PyInstaller at runtime
    and is not present during normal Python execution, which is why
    we need to check for its existence with hasattr().
    """
    # This is a special path created by PyInstaller at runtime.
    # We can check for its existence to see if we are running in a packaged app.
    if hasattr(sys, '_MEIPASS'):
        # Using getattr to avoid linter warnings about unknown attributes
        # _MEIPASS is added by PyInstaller at runtime
        return getattr(sys, '_MEIPASS')
    # For local testing of the hook
    return os.path.abspath(".")

# In the PyInstaller environment, sys._MEIPASS is the root
# The llama_cpp libraries are in _MEIPASS/llama_cpp/lib
meipass = get_meipass_path()
lib_dir = os.path.join(meipass, "llama_cpp", "lib")
logger.info(f"Calculated library directory: {lib_dir}")

if not os.path.isdir(lib_dir):
    logger.error(f"Library directory does not exist: {lib_dir}")
else:
    logger.info(f"Library directory found: {lib_dir}")
    # Add the library directory to the DLL search path
    try:
        os.add_dll_directory(lib_dir)
        logger.info(f"Successfully added to DLL search path: {lib_dir}")
        
        # List all DLLs in the directory to debug
        dlls = [os.path.basename(f) for f in glob.glob(os.path.join(lib_dir, "*.dll"))]
        logger.info(f"DLLs found in library directory: {dlls}")
    except Exception as e:
        logger.error(f"Failed to add to DLL search path: {e}")

    # Specifically add CUDA 12.9 directory to the search path
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
    if os.path.exists(cuda_path):
        try:
            os.add_dll_directory(cuda_path)
            logger.info(f"Added CUDA 12.9 path to DLL search: {cuda_path}")
            
            # Check if key CUDA DLLs exist here
            cuda_dlls = [
                os.path.basename(f) for f in glob.glob(os.path.join(cuda_path, "*.dll"))
            ]
            logger.info(f"Found {len(cuda_dlls)} CUDA DLLs in {cuda_path}")
            # Log the first few DLLs for debugging
            for dll in cuda_dlls[:10]:
                logger.info(f"  - {dll}")
                
            # Check specifically for critical DLLs
            critical_dlls = ["cudart64_12.dll", "cublas64_12.dll"]
            for dll in critical_dlls:
                if any(dll in d for d in cuda_dlls):
                    logger.info(f"  ✓ Found critical DLL: {dll}")
                else:
                    logger.warning(f"  ⚠ Missing critical DLL: {dll}")
        except Exception as e:
            logger.error(f"Failed to add CUDA path {cuda_path} to DLL search: {e}")
    else:
        logger.warning(f"CUDA 12.9 path not found: {cuda_path}")

    # Explicitly load the core DLLs in the correct order
    # ggml-cuda.dll should be loaded before llama.dll if CUDA is to be used
    dlls_to_load = ["ggml-cuda.dll", "llama.dll"]
    for dll in dlls_to_load:
        dll_path = os.path.join(lib_dir, dll)
        if os.path.exists(dll_path):
            try:
                ctypes.CDLL(dll_path)
                logger.info(f"Successfully pre-loaded DLL: {dll_path}")
            except Exception as e:
                logger.error(f"Failed to pre-load DLL {dll_path}: {e}")
                # Try to extract more details about the error
                if "cublas64" in str(e) or "cudart64" in str(e):
                    logger.error(f"CUDA DLL error detected! This indicates missing CUDA runtime libraries.")
                    logger.error(f"Please ensure CUDA 12.9 is installed on the system.")
        else:
            logger.warning(f"DLL not found for pre-loading: {dll_path}")

logger.info("Finished runtime hook: dll_load_hook.py")
