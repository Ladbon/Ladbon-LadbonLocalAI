"""
Special module to handle DLL loading in PyInstaller onefile mode
This module is imported early in the startup process to ensure
proper handling of the temporary directory and DLLs
"""
import os
import sys
import ctypes
import logging
import shutil
from datetime import datetime

# Import type stubs to silence IDE warnings about PyInstaller specific attributes
try:
    import pyinstaller_types  # type: ignore # noqa
except ImportError:
    pass  # Will be absent at runtime in bundle

# Set up logging
log_file = os.path.join(os.getcwd(), f"onefile_loader_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

def setup_logging():
    """Setup basic logging for this module"""
    logger = logging.getLogger("onefile_loader")
    logger.setLevel(logging.DEBUG)
    
    # Create file handler with UTF-8 encoding
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(fh)
    
    return logger

logger = setup_logging()

def prepare_temp_directory():
    """
    Prepare the PyInstaller temporary directory for llama-cpp
    This is called very early in the startup process
    """
    logger.info("Starting onefile loader preparation")
    
    try:
        # Check if we're running in a PyInstaller bundle
        if not hasattr(sys, "_MEIPASS"):
            logger.info("Not running in PyInstaller bundle, skipping")
            return
            
        # Get the PyInstaller bundle directory safely
        try:
            from pyinstaller_types import get_bundle_dir
            meipass_dir = get_bundle_dir()
        except ImportError:
            # Fall back if type stubs aren't available
            meipass_dir = sys._MEIPASS  # type: ignore # noqa
            
        logger.info(f"PyInstaller _MEIPASS directory: {meipass_dir}")
        
        # Create the llama_cpp/lib directory structure
        llama_cpp_dir = os.path.join(meipass_dir, "llama_cpp")
        lib_dir = os.path.join(llama_cpp_dir, "lib")
        
        logger.info(f"Creating directories: {llama_cpp_dir} and {lib_dir}")
        os.makedirs(lib_dir, exist_ok=True)
        
        # List all DLLs in the _MEIPASS directory
        dlls = [f for f in os.listdir(meipass_dir) if f.lower().endswith('.dll')]
        logger.info(f"Found {len(dlls)} DLLs in _MEIPASS directory")
        
        # Copy llama-cpp related DLLs to the lib directory
        for dll in dlls:
            if any(name in dll.lower() for name in ["ggml", "llama", "llava"]):
                src = os.path.join(meipass_dir, dll)
                dst = os.path.join(lib_dir, dll)
                logger.info(f"Copying {dll} to {dst}")
                shutil.copy2(src, dst)
        
        # On Windows, use AddDllDirectory to add the lib directory to the search path
        if os.name == 'nt':
            logger.info("Adding DLL directory to search path")
            try:
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                
                # Convert to wide string for Windows API
                lib_dir_c = ctypes.c_wchar_p(lib_dir)
                # Add the directory to the DLL search path
                handle = kernel32.AddDllDirectory(lib_dir_c)
                if handle:
                    logger.info(f"Successfully added DLL directory: {lib_dir}")
                else:
                    error = ctypes.get_last_error()
                    logger.warning(f"Failed to add DLL directory: {lib_dir}, error code: {error}")
            except Exception as e:
                logger.error(f"Error adding DLL directory: {str(e)}")
        
        # Also add to PATH environment variable as a fallback
        logger.info(f"Adding {lib_dir} to PATH environment")
        os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')
        
        logger.info("Onefile preparation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in prepare_temp_directory: {str(e)}")
        return False

# Always run the preparation when this module is imported
prepare_temp_directory()
