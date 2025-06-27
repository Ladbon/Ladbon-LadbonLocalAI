"""
This module provides utility functions to help load the llama-cpp library in bundled environments
"""
import os
import sys
import logging
import importlib

logger = logging.getLogger('llama_cpp_loader')

def setup_llamacpp_environment():
    """
    Sets up the environment for loading llama-cpp-python
    This is necessary when running from a PyInstaller bundle
    """
    try:
        # Check if we're in a PyInstaller bundle
        if getattr(sys, '_MEIPASS', None):
            # Add the lib directory to PATH
            bundle_dir = sys._MEIPASS  # type: ignore # PyInstaller adds this at runtime
            logger.info(f"Running from PyInstaller bundle: {bundle_dir}")
            
            # For standalone exe, ensure the directory structure exists
            # This is needed because the onefile mode extracts to a temp directory
            llama_cpp_lib_dir = os.path.join(bundle_dir, 'llama_cpp', 'lib')
            
            # Create the directory structure if it doesn't exist
            if not os.path.exists(llama_cpp_lib_dir):
                logger.info(f"Creating missing llama_cpp lib directory at: {llama_cpp_lib_dir}")
                try:
                    os.makedirs(llama_cpp_lib_dir, exist_ok=True)
                    
                    # Look for DLLs in the root directory and copy them to the lib directory
                    dll_files = [f for f in os.listdir(bundle_dir) if f.lower().endswith('.dll')]
                    for dll in dll_files:
                        if any(dll_name in dll.lower() for dll_name in ['ggml', 'llama', 'llava']):
                            src_path = os.path.join(bundle_dir, dll)
                            dst_path = os.path.join(llama_cpp_lib_dir, dll)
                            logger.info(f"Copying {dll} to lib directory")
                            try:
                                import shutil
                                shutil.copy2(src_path, dst_path)
                            except Exception as copy_err:
                                logger.warning(f"Error copying {dll}: {copy_err}")
                except Exception as dir_err:
                    logger.error(f"Error creating lib directory: {dir_err}")
            
            # Try multiple potential locations for the lib directory
            potential_paths = [
                os.path.join(bundle_dir, 'llama_cpp', 'lib'),
                os.path.join(bundle_dir, 'lib'),
                bundle_dir
            ]
            
            lib_found = False
            for lib_path in potential_paths:
                if os.path.exists(lib_path):
                    logger.info(f"Found potential library directory at: {lib_path}")
                    
                    # Add to PATH so DLLs can be found
                    os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
                    logger.info(f"Added {lib_path} to PATH")
                    
                    # List available files
                    try:
                        files = os.listdir(lib_path)
                        logger.info(f"Files in {lib_path}: {', '.join(files) if files else 'No files found'}")
                        
                        # Check if we have any .dll files
                        dll_files = [f for f in files if f.lower().endswith('.dll')]
                        if dll_files:
                            logger.info(f"Found {len(dll_files)} DLL files in {lib_path}")
                            lib_found = True
                    except Exception as list_err:
                        logger.warning(f"Error listing files in {lib_path}: {list_err}")
            
            if lib_found:
                return True
            else:
                logger.warning("No llama_cpp library directory with DLLs was found")
        else:
            logger.info("Not running from PyInstaller bundle, using standard library paths")
            # Try to use llama_cpp.lib_path if available
            try:
                from llama_cpp.lib_path import get_lib_path
                lib_path = get_lib_path()
                logger.info(f"Using lib path from llama_cpp.lib_path: {lib_path}")
                if os.path.exists(lib_path):
                    os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
                    return True
            except ImportError:
                logger.warning("llama_cpp.lib_path module not found")
        
        # If we get here, we couldn't find a suitable lib directory
        # Let's try to find any llama_cpp installation
        try:
            import llama_cpp
            logger.info(f"llama_cpp found at {llama_cpp.__file__}")
            llamacpp_dir = os.path.dirname(llama_cpp.__file__)
            logger.info(f"Checking for lib directory in {llamacpp_dir}")
            lib_path = os.path.join(llamacpp_dir, "lib")
            if os.path.exists(lib_path):
                os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
                logger.info(f"Added {lib_path} to PATH")
                return True
        except ImportError:
            logger.warning("llama_cpp module not found")
        
        return False
    except Exception as e:
        logger.error(f"Error setting up llama_cpp environment: {str(e)}", exc_info=True)
        return False

def initialize_llamacpp():
    """
    Initialize llama-cpp-python with proper error handling
    Returns True if successful, False otherwise
    """
    try:
        import llama_cpp
        import ctypes
        
        # Patch the library initialization
        logger.info("Initializing llama-cpp backend...")
        
        # Find the library in either direct _lib or nested module structure
        lib = None
        if hasattr(llama_cpp, '_lib') and hasattr(llama_cpp.llama_cpp._lib, 'llama_backend_init'):
            lib = llama_cpp.llama_cpp._lib
            logger.info("Found backend init in llama_cpp._lib")
        elif hasattr(llama_cpp, 'llama_cpp') and hasattr(llama_cpp.llama_cpp, '_lib') and hasattr(llama_cpp.llama_cpp._lib, 'llama_backend_init'):
            lib = llama_cpp.llama_cpp._lib
            logger.info("Found backend init in llama_cpp.llama_cpp._lib")
        else:
            # Try importing directly as a last resort
            try:
                lib = importlib.import_module("llama_cpp.llama_cpp")._lib
                logger.info("Imported llama_cpp.llama_cpp._lib directly")
            except Exception as import_err:
                logger.warning(f"Could not find llama_backend_init in any known location: {str(import_err)}")
                return False
        
        # Set up function signature
        lib.llama_backend_init.argtypes = [ctypes.c_bool]
        lib.llama_backend_init.restype = None
        
        # Define and patch the backend_init function
        def backend_init(numa: bool = False):
            return lib.llama_backend_init(ctypes.c_bool(numa))
        
        llama_cpp.llama_backend_init = backend_init
        
        # Test initialization
        logger.info("Calling llama_backend_init...")
        llama_cpp.llama_backend_init(False)
        logger.info("llama-cpp backend initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize llama-cpp: {str(e)}", exc_info=True)
        return False
