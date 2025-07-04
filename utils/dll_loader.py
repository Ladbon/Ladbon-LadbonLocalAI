"""
Utility module to help locate and load DLLs for llama-cpp-python when running from a packaged executable
This is necessary because PyInstaller sometimes has issues with locating the correct DLLs
"""
import os
import sys
import ctypes
import logging
import platform

logger = logging.getLogger('dll_loader')

def find_dlls(dll_name, base_paths=None):
    """
    Find DLLs in various possible locations, prioritizing the PyInstaller _internal path
    """
    if base_paths is None:
        base_paths = []
    
    # For frozen apps, prioritize the PyInstaller _internal path
    if getattr(sys, 'frozen', False):
        # PyInstaller specific paths - prioritize these
        base_paths.append(os.path.join(os.path.dirname(sys.executable), '_internal', 'llama_cpp', 'lib'))
        base_paths.append(os.path.join(os.path.dirname(sys.executable), '_internal', 'llama_cpp'))
        base_paths.append(os.path.join(os.path.dirname(sys.executable), '_internal'))
    
    # Add the executable directory last (lowest priority)
    base_paths.append(os.path.dirname(sys.executable))
    
    # Add current directory and its subdirectories
    base_paths.append(os.getcwd())
    base_paths.append(os.path.join(os.getcwd(), 'llama_cpp', 'lib'))
    base_paths.append(os.path.join(os.getcwd(), 'llama_cpp'))
    
    # Add explicit Python paths
    try:
        import llama_cpp
        llama_cpp_dir = os.path.dirname(llama_cpp.__file__)
        base_paths.append(llama_cpp_dir)
        base_paths.append(os.path.join(llama_cpp_dir, 'lib'))
    except ImportError:
        pass
    
    # Remove duplicates while preserving order
    seen = set()
    base_paths = [p for p in base_paths if not (p in seen or seen.add(p))]
    
    # Log the search paths
    logger.info(f"Searching for {dll_name} in these directories:")
    for path in base_paths:
        logger.info(f"  - {path}")
    
    # Find all matching DLLs
    found_dlls = []
    for base_path in base_paths:
        if not os.path.isdir(base_path):
            continue
        
        dll_path = os.path.join(base_path, dll_name)
        if os.path.isfile(dll_path):
            found_dlls.append(dll_path)
            logger.info(f"Found DLL at: {dll_path}")
    
    if not found_dlls:
        logger.warning(f"No {dll_name} found in any of the search paths")
    
    return found_dlls

def load_llama_cpp_dlls():
    """
    Attempt to load all necessary DLLs for llama-cpp-python
    """
    if platform.system() != "Windows":
        logger.info("DLL loading only needed on Windows - skipping")
        return True
    
    # On Windows, try to load the DLLs in the correct order
    # First try loading llama.dll, then GGML libraries
    dll_names = ['llama.dll', 'ggml-base.dll', 'ggml-cpu.dll']
    
    # Check system architecture
    is_64bit = sys.maxsize > 2**32
    logger.info(f"Python interpreter architecture: {'64-bit' if is_64bit else '32-bit'}")
    if not is_64bit:
        logger.error("CRITICAL: Running in 32-bit Python! llama-cpp-python requires 64-bit Python")
        return False
    
    # Get the directory of the application to find potential DLL paths
    if getattr(sys, 'frozen', False):
        app_path = os.path.dirname(sys.executable)
        logger.info(f"Running as frozen application from: {app_path}")
    else:
        app_path = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Running as script from: {app_path}")
    
    success = True
    loaded_dlls = []
    
    for dll_name in dll_names:
        dll_paths = find_dlls(dll_name)
        if not dll_paths:
            logger.error(f"Critical DLL {dll_name} not found")
            success = False
            continue
        
        # Try loading each found DLL until one works
        dll_loaded = False
        for dll_path in dll_paths:
            try:
                dll = ctypes.CDLL(dll_path)
                loaded_dlls.append(dll)
                logger.info(f"Successfully loaded: {dll_path}")
                dll_loaded = True
                break
            except Exception as e:
                logger.warning(f"Failed to load {dll_path}: {e}")
        
        if not dll_loaded:
            logger.error(f"Could not load any instance of {dll_name}")
            success = False
    
    return success

def ensure_dlls_loadable():
    """
    Main function to ensure DLLs are available and loaded
    """
    try:
        # Initialize the logger
        logging.basicConfig(level=logging.INFO)
        
        # Try loading the DLLs
        if load_llama_cpp_dlls():
            logger.info("Successfully loaded all required DLLs")
            return True
        else:
            logger.error("Failed to load all required DLLs")
            return False
    except Exception as e:
        logger.error(f"Error in DLL loading process: {e}")
        return False

if __name__ == "__main__":
    ensure_dlls_loadable()
