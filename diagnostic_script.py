"""
Enhanced diagnostics script for llama-cpp-python DLL loading issues.
Run this script from within the packaged app directory to check for DLL loading issues.

This script:
1. Lists all DLLs in the package
2. Tries to load llama.dll directly 
3. Inspects dependencies of critical DLLs
4. Checks if CUDA DLLs are properly loaded
"""

import os
import sys
import ctypes
import platform
from ctypes import WinDLL
import logging
import glob
import traceback
import time

# Add venv site-packages to path
venv_path = os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)
    print(f"INFO - Added to path: {venv_path}")
else:
    print(f"WARNING - venv site-packages not found at: {venv_path}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("dll_diagnostics.log", mode='w')
    ]
)

logger = logging.getLogger("dll_diagnostics")

def list_dlls_in_directory(directory):
    """List all DLLs in a directory and subdirectories"""
    logger.info(f"Scanning for DLLs in {directory}")
    dlls = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.dll'):
                full_path = os.path.join(root, file)
                dlls.append(full_path)
    
    # Sort by filename for easier reading
    dlls.sort(key=lambda x: os.path.basename(x).lower())
    
    # Print all found DLLs
    logger.info(f"Found {len(dlls)} DLLs:")
    for dll in dlls:
        logger.info(f"  {os.path.basename(dll)} - {dll}")
    
    return dlls

def check_llama_cpp_files():
    """Check if llama_cpp files exist and list them"""
    # First try to find where llama_cpp is in the package
    possible_locations = [
        os.path.join(os.getcwd(), "_internal", "llama_cpp"),
        os.path.join(os.getcwd(), "llama_cpp"),
        os.path.join(os.path.dirname(os.getcwd()), "_internal", "llama_cpp"),
        os.path.join(os.path.dirname(os.getcwd()), "llama_cpp")
    ]
    
    llama_cpp_dir = None
    for loc in possible_locations:
        if os.path.exists(loc):
            llama_cpp_dir = loc
            logger.info(f"Found llama_cpp directory at: {llama_cpp_dir}")
            break
    
    if not llama_cpp_dir:
        logger.error("Could not find llama_cpp directory!")
        return None
        
    # List files in the directory
    logger.info(f"Files in {llama_cpp_dir}:")
    for file in os.listdir(llama_cpp_dir):
        logger.info(f"  {file}")
    
    # Check for lib directory
    lib_dir = os.path.join(llama_cpp_dir, "lib")
    if os.path.exists(lib_dir):
        logger.info(f"Found lib directory at: {lib_dir}")
        logger.info(f"Files in {lib_dir}:")
        for file in os.listdir(lib_dir):
            logger.info(f"  {file}")
        
        # Return the lib directory for further analysis
        return lib_dir
    else:
        logger.error(f"No lib directory found at {lib_dir}")
        return None

def try_load_dll(dll_path):
    """Try to load a DLL directly using ctypes"""
    logger.info(f"Trying to load DLL: {dll_path}")
    try:
        dll = WinDLL(dll_path)
        logger.info(f"Successfully loaded {os.path.basename(dll_path)}")
        return dll
    except Exception as e:
        logger.error(f"Failed to load {os.path.basename(dll_path)}: {e}")
        return None

def check_dll_dependencies(dll_path):
    """Check dependencies of a DLL using the Windows API"""
    if platform.system() != 'Windows':
        logger.warning("This function only works on Windows")
        return
        
    # Initialize win32api to None, we'll check if it's None before using it
    # Import win32api conditionally
    win32api_module = None
    try:
        # Only try to import if needed, but handle the case where it's not available
        import importlib.util
        if importlib.util.find_spec("win32api"):
            try:
                # Use a conditional import that works with type checkers
                from typing import Any
                import sys
                if "win32api" in sys.modules:
                    win32api_module = sys.modules["win32api"]
                else:
                    # Use exec to prevent Pylance from directly analyzing the import
                    module_name = "win32api"
                    exec(f"import {module_name} as temp_module")
                    win32api_module = locals()["temp_module"]
                logger.info("Successfully imported win32api for detailed DLL checks")
            except ImportError:
                logger.warning("Could not import win32api module despite it being found")
                win32api_module = None
        else:
            logger.warning("win32api module not found, can't check detailed DLL dependencies")
            logger.warning("Consider installing pywin32 with: pip install pywin32")
            # Continue with basic checks that don't require win32api
    except ImportError:
        logger.warning("win32api module not found, can't check detailed DLL dependencies")
        logger.warning("Consider installing pywin32 with: pip install pywin32")
        # Continue with basic checks that don't require win32api
    except Exception as import_err:
        logger.warning(f"Error importing win32api: {import_err}")
        # Continue with basic checks that don't require win32api
    
    try:
        logger.info(f"Checking dependencies for: {os.path.basename(dll_path)}")
        
        # Check if we have win32api available and use it if we do
        if win32api_module:
            try:
                # Use ctypes to get more information about the DLL
                logger.info(f"Using ctypes for basic DLL info for: {dll_path}")
                dll_handle = ctypes.WinDLL(dll_path)
                logger.info(f"Successfully loaded DLL with ctypes: {dll_path}")
                
                # Try to get file version info using win32api if available
                file_info = None
                try:
                    # Check if the required method exists
                    if hasattr(win32api_module, 'GetFileVersionInfo'):
                        file_info = win32api_module.GetFileVersionInfo(dll_path, "\\")
                    else:
                        logger.warning("GetFileVersionInfo method not found in win32api module")
                except Exception as ver_err:
                    logger.warning(f"Could not get file version info: {ver_err}")
                    file_info = None
                if file_info:
                    logger.info(f"File version info: {file_info}")
            except Exception as w32_err:
                logger.warning(f"Error getting detailed DLL info: {w32_err}")
        else:
            logger.info(f"Basic DLL info for: {dll_path} (win32api not available for detailed inspection)")
        
        # Try to load the DLL to see if it loads properly
        try:
            handle = ctypes.WinDLL(dll_path)
            logger.info(f"Successfully loaded DLL: {os.path.basename(dll_path)}")
        except Exception as load_err:
            logger.error(f"Failed to load DLL: {load_err}")
            
        # Check if file exists and its size
        if os.path.exists(dll_path):
            size = os.path.getsize(dll_path)
            logger.info(f"DLL file size: {size:,} bytes")
            logger.info(f"DLL last modified: {time.ctime(os.path.getmtime(dll_path))}")
        
        # For CUDA DLLs specifically, try to check version information
        if "cuda" in dll_path.lower() or "cublas" in dll_path.lower():
            logger.info("Attempting to get CUDA DLL version info...")
            # Try to extract version from filename
            filename = os.path.basename(dll_path)
            if "_" in filename:
                try:
                    # Extract version number from filename like cudart64_12.dll
                    parts = filename.split("_")
                    if len(parts) > 1:
                        version_part = parts[-1].split(".")[0]  # Get "12" from "cudart64_12.dll"
                        logger.info(f"CUDA DLL version (from filename): {version_part}")
                except Exception as ver_err:
                    logger.info(f"Could not extract version from filename: {ver_err}")
                    
            # Try using ctypes to get file version info (more generic, doesn't need win32api)
            try:
                dll_handle = ctypes.WinDLL(dll_path)
                if hasattr(dll_handle, 'cudaRuntimeGetVersion'):
                    version_ptr = ctypes.c_int()
                    result = dll_handle.cudaRuntimeGetVersion(ctypes.byref(version_ptr))
                    if result == 0:  # CUDA_SUCCESS
                        version = version_ptr.value
                        major = version // 1000
                        minor = (version % 1000) // 10
                        logger.info(f"CUDA runtime version: {major}.{minor}")
                    else:
                        logger.info(f"Failed to get CUDA version, error code: {result}")
            except Exception as ver_err:
                logger.info(f"Could not get detailed version info: {ver_err}")
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        logger.error(traceback.format_exc())

def try_import_llama_cpp():
    """Try importing llama_cpp and check basic functionality"""
    logger.info("Attempting to import llama_cpp...")
    try:
        # Set environment variables to force CPU-only mode
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["FORCE_CPU_ONLY"] = "1"
        
        import llama_cpp
        logger.info(f"Successfully imported llama_cpp from {llama_cpp.__file__}")
        logger.info(f"llama_cpp version: {getattr(llama_cpp, '__version__', 'Unknown')}")
        
        # Check library directory
        lib_dir = os.path.join(os.path.dirname(llama_cpp.__file__), 'lib')
        if os.path.exists(lib_dir):
            logger.info(f"Library directory exists at {lib_dir}")
            libs = os.listdir(lib_dir)
            logger.info(f"Found {len(libs)} files in lib directory: {libs}")
            
            # Try loading the llama.dll directly
            llama_dll_path = os.path.join(lib_dir, "llama.dll")
            if os.path.exists(llama_dll_path):
                try:
                    dll = WinDLL(llama_dll_path)
                    logger.info(f"Successfully loaded llama.dll directly from {llama_dll_path}")
                except Exception as dll_err:
                    logger.error(f"Failed to load llama.dll directly: {dll_err}")
        else:
            logger.warning(f"Library directory not found at {lib_dir}")
            
        # Log all available attributes
        logger.info(f"llama_cpp module attributes: {dir(llama_cpp)}")
        
        # Check if we can initialize the backend with CPU-only mode
        logger.info("Attempting to initialize llama_cpp backend in CPU-only mode...")
        try:                # Check if backend_init is available
            if hasattr(llama_cpp, 'llama_backend_init'):
                logger.info("Found llama_backend_init function directly in llama_cpp module")
                try:
                    logger.info("Calling llama_backend_init...")
                    # Check if it's a function or an attribute
                    if callable(llama_cpp.llama_backend_init):
                        # Check function signature to see if it takes arguments
                        try:
                            import inspect
                            sig = inspect.signature(llama_cpp.llama_backend_init)
                            params = list(sig.parameters.keys())
                            logger.info(f"llama_backend_init takes parameters: {params}")
                            
                            # Try to call it based on signature
                            logger.info("Attempting to call backend_init based on signature...")
                            try:
                                # First try to look at the signature
                                if params:
                                    # Try with kwargs if parameters exist
                                    logger.info(f"Calling with parameters: {params}")
                                    # Create dynamic kwargs based on first parameter
                                    param_name = params[0]
                                    kwargs = {param_name: False}
                                    try:
                                        llama_cpp.llama_backend_init(**kwargs)
                                    except Exception:
                                        # If kwargs failed, try without parameters
                                        logger.info("Kwargs approach failed, trying without params")
                                        llama_cpp.llama_backend_init()
                                else:
                                    # Try without parameters
                                    logger.info("Calling without parameters")
                                    llama_cpp.llama_backend_init()
                            except TypeError as e:
                                logger.warning(f"TypeError calling backend_init: {e}")
                                # If one way failed, try the other
                                if "takes 0 positional arguments" in str(e):
                                    logger.info("Retrying without arguments")
                                    llama_cpp.llama_backend_init()
                                elif "missing 1 required positional argument" in str(e) or "unexpected keyword argument" in str(e):
                                    # Try a different approach
                                    logger.info("Type error occurred, trying alternate initialization")
                                    try:
                                        # Try to directly access the C function
                                        if hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib"):
                                            logger.info("Attempting to call backend_init via _lib...")
                                            llama_cpp.llama_cpp._lib.llama_backend_init()
                                    except Exception as e3:
                                        logger.warning(f"All initialization attempts failed: {e3}")
                                        logger.info("Continuing without initialization")
                        except Exception as sig_err:
                            logger.warning(f"Could not inspect function signature: {sig_err}")
                            # Try calling without arguments as fallback
                            llama_cpp.llama_backend_init()
                    else:
                        # It might be a property or other attribute
                        logger.info(f"llama_backend_init is not callable: {type(llama_cpp.llama_backend_init)}")
                    logger.info("Successfully initialized backend!")
                except Exception as init_err:
                    logger.error(f"Error calling llama_backend_init: {init_err}")
                    logger.error(traceback.format_exc())
                    logger.error("This is the access violation error we need to diagnose")
            else:
                logger.warning("llama_backend_init function not found directly in llama_cpp module")
                
                # Check for nested module structures
                backend_init_found = False
                
                # Try to access potential internal libraries
                for attr_name in dir(llama_cpp):
                    if attr_name.startswith('_') and 'lib' in attr_name.lower():
                        logger.info(f"Found potential library attribute: {attr_name}")
                        try:
                            lib = getattr(llama_cpp, attr_name)
                            logger.info(f"Successfully accessed {attr_name}")
                            
                            # Check for backend_init function
                            if hasattr(lib, 'llama_backend_init'):
                                logger.info(f"Found llama_backend_init in llama_cpp.{attr_name}")
                                backend_init_found = True
                                try:
                                    # Try calling it with safer approach
                                    logger.info(f"Attempting to call llama_backend_init via {attr_name}...")
                                    # We'll use the existing ctypes module rather than trying to 
                                    # set argtypes and restype
                                    backend_init_func = getattr(lib, 'llama_backend_init')
                                    backend_init_func(False)
                                    logger.info(f"Successfully initialized backend through {attr_name}!")
                                except Exception as lib_err:
                                    logger.error(f"Error calling backend_init via {attr_name}: {lib_err}")
                                    logger.error(traceback.format_exc())
                        except Exception as access_err:
                            logger.error(f"Error accessing {attr_name}: {access_err}")
                
                # Try llama_cpp.llama_cpp (nested module)
                if hasattr(llama_cpp, 'llama_cpp'):
                    logger.info("Found nested llama_cpp.llama_cpp module")
                    nested_module = llama_cpp.llama_cpp
                    
                    # Look for library-like attributes in the nested module
                    for attr_name in dir(nested_module):
                        if attr_name.startswith('_') and 'lib' in attr_name.lower():
                            logger.info(f"Found potential library attribute in nested module: {attr_name}")
                            try:
                                lib = getattr(nested_module, attr_name)
                                logger.info(f"Successfully accessed llama_cpp.llama_cpp.{attr_name}")
                                
                                # Check for backend_init function
                                if hasattr(lib, 'llama_backend_init'):
                                    logger.info(f"Found llama_backend_init in llama_cpp.llama_cpp.{attr_name}")
                                    backend_init_found = True
                                    try:
                                        # Try calling it
                                        logger.info(f"Attempting to call llama_backend_init via nested module {attr_name}...")
                                        backend_init_func = getattr(lib, 'llama_backend_init')
                                        backend_init_func(False)
                                        logger.info(f"Successfully initialized backend through nested module!")
                                    except Exception as nested_err:
                                        logger.error(f"Error calling backend_init via nested module: {nested_err}")
                                        logger.error(traceback.format_exc())
                            except Exception as access_err:
                                logger.error(f"Error accessing {attr_name} in nested module: {access_err}")
                
                if not backend_init_found:
                    logger.error("Could not find llama_backend_init in any expected location!")
                        
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            logger.error(traceback.format_exc())
            
        return True
    except ImportError as e:
        logger.error(f"Failed to import llama_cpp: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error importing llama_cpp: {e}")
        logger.error(traceback.format_exc())
        return False

def test_create_llama_instance():
    """Test creating a Llama instance with a small model if available"""
    logger.info("Checking for a test model...")
    model_paths = []
    
    # Check common model locations
    check_dirs = [
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.path.dirname(os.getcwd()), "models"),
        os.path.expanduser("~/Ladbon AI Desktop/models"),
        os.path.expanduser("~/.ladbon_ai/models")
    ]
    
    for dir_path in check_dirs:
        if os.path.exists(dir_path):
            logger.info(f"Checking for models in {dir_path}")
            for file in os.listdir(dir_path):
                if file.endswith('.gguf') and ("tiny" in file.lower() or "small" in file.lower()):
                    model_path = os.path.join(dir_path, file)
                    model_paths.append(model_path)
                    logger.info(f"Found potential test model: {file}")
    
    if not model_paths:
        logger.warning("No suitable test models found")
        return
    
    # Try to create a Llama instance with the first model
    try:
        logger.info(f"Trying to load model: {model_paths[0]}")
        # Force environment variables for CPU-only mode
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["DISABLE_CUDA"] = "1"
        os.environ["FORCE_CPU_ONLY"] = "1"
        
        # Import llama_cpp and create Llama instance
        import llama_cpp
        from llama_cpp import Llama
        params = {
            "model_path": model_paths[0],
            "n_ctx": 512,  # Small context to minimize memory
            "n_batch": 512,
            "verbose": True,
            "n_gpu_layers": 0  # Force CPU-only mode
        }
        logger.info(f"Creating Llama instance with params: {params}")
        model = Llama(**params)
        logger.info("Successfully created Llama instance!")
        
        # Try a simple completion
        logger.info("Testing model with a simple prompt...")
        result = model.create_completion("Hello, world!", max_tokens=10)
        logger.info(f"Model response: {result}")
        
    except Exception as e:
        logger.error(f"Failed to create Llama instance: {e}")
        logger.error(traceback.format_exc())

def main():
    """Main diagnostic function"""
    logger.info("=== DLL Diagnostics Tool ===")
    logger.info(f"Running on Python {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"PATH: {os.environ.get('PATH', 'Not set')}")
    
    # Step 1: List DLLs in the current directory and subdirectories
    dlls = list_dlls_in_directory(os.getcwd())
    
    # Step 2: Check for llama_cpp files
    lib_dir = check_llama_cpp_files()
    
    # Step 3: Try to load specific DLLs
    if lib_dir:
        llama_dll_path = os.path.join(lib_dir, "llama.dll")
        if os.path.exists(llama_dll_path):
            llama_dll = try_load_dll(llama_dll_path)
            if llama_dll:
                # Check dependencies
                check_dll_dependencies(llama_dll_path)
                
        # Check CUDA DLLs if present
        cuda_dlls = [f for f in os.listdir(lib_dir) if "cuda" in f.lower()]
        for cuda_dll in cuda_dlls:
            dll_path = os.path.join(lib_dir, cuda_dll)
            try_load_dll(dll_path)
            check_dll_dependencies(dll_path)
    
    # Step 4: Try to import llama_cpp
    if try_import_llama_cpp():
        # Step 5: Try to create a Llama instance
        test_create_llama_instance()
    
    logger.info("=== Diagnostics Complete ===")
    
if __name__ == "__main__":
    main()
    input("Press Enter to exit...")
