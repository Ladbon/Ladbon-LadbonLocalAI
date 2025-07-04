"""
Explicit DLL pre-loader to be called at application startup
This ensures llama-cpp-python DLLs are properly loaded before being used
"""
import os
import sys
import ctypes
import platform
import logging
import glob
from pathlib import Path

logger = logging.getLogger("dll_preloader")

def find_dlls(dll_names):
    """
    Find all DLLs that match the given names in various possible locations
    """
    if not isinstance(dll_names, list):
        dll_names = [dll_names]
    
    # Get the directory of the executable
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # We're running as a bundled exe
        base_path = sys.executable
        exe_dir = os.path.dirname(base_path)
        internal_dir = os.path.join(exe_dir, '_internal')
    else:
        # We're running in a normal Python environment
        base_path = sys.executable
        exe_dir = os.path.dirname(base_path)
        internal_dir = os.path.join(os.path.dirname(__file__))
    
    # Potential DLL locations
    dll_locations = [
        exe_dir,
        internal_dir,
        os.path.join(internal_dir, 'llama_cpp'),
        os.path.join(internal_dir, 'llama_cpp', 'lib'),
        os.path.join(exe_dir, 'llama_cpp'),
        os.path.join(exe_dir, 'llama_cpp', 'lib'),
        os.path.join(os.getcwd())
    ]
    
    # Try to import llama_cpp to find its location
    try:
        import llama_cpp
        llama_cpp_path = os.path.dirname(llama_cpp.__file__)
        dll_locations.append(llama_cpp_path)
        dll_locations.append(os.path.join(llama_cpp_path, 'lib'))
    except:
        pass
    
    # Find all matching DLLs
    found_dlls = {}
    for dll_name in dll_names:
        found_dlls[dll_name] = []
        
        for location in dll_locations:
            if not os.path.exists(location):
                continue
            
            # Try exact match
            dll_path = os.path.join(location, dll_name)
            if os.path.exists(dll_path):
                found_dlls[dll_name].append(dll_path)
                continue
            
            # Try wildcard match
            if '*' in dll_name:
                pattern = os.path.join(location, dll_name)
                for match in glob.glob(pattern):
                    found_dlls[dll_name].append(match)
    
    return found_dlls

def preload_llama_dlls():
    """
    Pre-load all necessary llama-cpp DLLs
    """
    if platform.system() != "Windows":
        logger.info("DLL preloading only needed on Windows")
        return True
    
    # DLLs to preload
    dll_names = [
        'llama.dll',
        'ggml-*.dll',
        'avx*.dll',
        'fmha*.dll'
    ]
    
    # Find all matching DLLs
    found_dlls = find_dlls(dll_names)
    
    # Log what we found
    logger.info("Found DLLs:")
    for dll_name, paths in found_dlls.items():
        if paths:
            logger.info(f"  {dll_name}: {len(paths)} found")
            for path in paths:
                logger.info(f"    - {path}")
        else:
            logger.warning(f"  {dll_name}: Not found")
    
    # Load all found DLLs
    loaded_dlls = {}
    for dll_name, paths in found_dlls.items():
        loaded_dlls[dll_name] = []
        for path in paths:
            try:
                dll = ctypes.CDLL(path)
                loaded_dlls[dll_name].append((path, dll))
                logger.info(f"Successfully loaded DLL: {path}")
            except Exception as e:
                logger.error(f"Failed to load DLL {path}: {e}")
    
    # Check if any DLLs were loaded
    loaded_count = sum(len(loaded) for loaded in loaded_dlls.values())
    if loaded_count == 0:
        logger.error("No DLLs were successfully loaded")
        return False
    
    logger.info(f"Successfully preloaded {loaded_count} DLLs")
    return True

def initialize():
    """
    Initialize DLL preloading
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Preload llama DLLs
    success = preload_llama_dlls()
    
    return success

if __name__ == "__main__":
    initialize()
