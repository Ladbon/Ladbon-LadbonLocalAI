"""
This module provides utility functions to help load the llama-cpp library in bundled environments
"""
import os
import sys
import logging
import importlib
import ctypes
import ctypes.util
import glob
import platform
from ctypes import WinDLL, c_void_p
import time

logger = logging.getLogger('llama_cpp_loader')

def check_cuda_availability():
    """
    Check if CUDA is available and log relevant information
    Returns True if CUDA appears to be available
    """
    logger.info("=== CUDA DIAGNOSTICS ===")
    cuda_available = False
    
    # Check for CUDA DLLs in PATH
    try:
        logger.info(f"Current PATH: {os.environ.get('PATH', 'Not set')}")
        
        # Check for CUDA in environment variables
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            logger.info(f"CUDA_HOME/CUDA_PATH is set to: {cuda_home}")
            cuda_bin = os.path.join(cuda_home, "bin")
            if os.path.exists(cuda_bin):
                logger.info(f"CUDA bin directory exists: {cuda_bin}")
                # List CUDA DLLs
                cuda_dlls = glob.glob(os.path.join(cuda_bin, "*.dll"))
                logger.info(f"Found {len(cuda_dlls)} DLLs in CUDA bin directory")
                for dll in cuda_dlls[:5]:  # Show only first 5
                    logger.info(f"  - {os.path.basename(dll)}")
                if len(cuda_dlls) > 5:
                    logger.info(f"  - ... and {len(cuda_dlls) - 5} more")
                cuda_available = len(cuda_dlls) > 0
        else:
            logger.info("CUDA_HOME/CUDA_PATH is not set in environment")
        
        # Search for CUDA DLLs in PATH
        paths = os.environ.get("PATH", "").split(os.pathsep)
        cuda_in_path = False
        
        # Known CUDA paths
        known_paths = []
        if platform.system() == "Windows":
            known_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                r"C:\Program Files\NVIDIA Corporation"
            ]
            
            # Find all CUDA version directories
            for base_path in known_paths:
                if os.path.exists(base_path):
                    for cuda_ver in os.listdir(base_path):
                        cuda_dir = os.path.join(base_path, cuda_ver)
                        if os.path.isdir(cuda_dir) and cuda_ver.startswith("v"):
                            bin_path = os.path.join(cuda_dir, "bin")
                            if os.path.exists(bin_path) and os.path.isdir(bin_path):
                                logger.info(f"Found CUDA directory: {bin_path}")
                                # Check if it's in PATH
                                if bin_path in paths:
                                    logger.info(f"CUDA {cuda_ver} bin directory is in PATH")
                                    cuda_in_path = True
                                    cuda_available = True
                                else:
                                    logger.info(f"CUDA {cuda_ver} bin directory is NOT in PATH")
                                
                                # List the cudart DLLs
                                cudart_dlls = glob.glob(os.path.join(bin_path, "cudart64_*.dll"))
                                for dll in cudart_dlls:
                                    logger.info(f"Found CUDA runtime: {os.path.basename(dll)}")
                                
                                # Try loading the cudart DLL
                                for dll in cudart_dlls:
                                    try:
                                        WinDLL(dll)
                                        logger.info(f"Successfully loaded {os.path.basename(dll)}")
                                    except Exception as e:
                                        logger.warning(f"Failed to load {os.path.basename(dll)}: {str(e)}")
        
        # Try loading standard CUDA DLLs via find_library
        cuda_libs = [
            "cudart", "cudart64_120", "cudart64_121", "cudart64_122", "cudart64_123", "cudart64_124", 
            "cudart64_125", "cudart64_126", "cudart64_127", "cudart64_128", "cudart64_129", "cudart64_12",
            "cublas64_12", "cublasLt64_12"
        ]
        
        for lib in cuda_libs:
            lib_path = ctypes.util.find_library(lib)
            if lib_path:
                logger.info(f"Found {lib} at: {lib_path}")
                # Try loading it
                try:
                    WinDLL(lib_path)
                    logger.info(f"Successfully loaded {lib}")
                    if "cudart" in lib:
                        cuda_available = True
                except Exception as e:
                    logger.warning(f"Failed to load {lib}: {str(e)}")
            else:
                logger.info(f"Could not find {lib} with find_library")
        
        # Check if NVIDIA driver is present
        try:
            nvidia_dll = ctypes.util.find_library("nvcuda")
            if nvidia_dll:
                logger.info(f"Found NVIDIA driver DLL at: {nvidia_dll}")
                try:
                    WinDLL(nvidia_dll)
                    logger.info("Successfully loaded NVIDIA driver DLL")
                    cuda_available = True
                except Exception as e:
                    logger.warning(f"Failed to load NVIDIA driver DLL: {str(e)}")
            else:
                logger.warning("Could not find NVIDIA driver DLL")
        except Exception as e:
            logger.warning(f"Error checking for NVIDIA driver: {str(e)}")
            
        # Log overall CUDA availability
        if cuda_available:
            logger.info("CUDA appears to be available on this system")
        else:
            logger.warning("CUDA does not appear to be available on this system")
        
        logger.info("=== END CUDA DIAGNOSTICS ===")
        return cuda_available
    except Exception as e:
        logger.error(f"Error checking CUDA availability: {str(e)}", exc_info=True)
        return False

def setup_llamacpp_environment(force_cpu=False):
    """
    Sets up the environment for loading llama-cpp-python
    This is necessary when running from a PyInstaller bundle
    
    Args:
        force_cpu: If True, forces CPU-only mode regardless of CUDA availability
    """
    try:
        # Check for CPU-only mode environment variables
        cpu_only = force_cpu or os.environ.get('FORCE_CPU_ONLY') == '1'
        if cpu_only:
            logger.info("CPU-only mode requested, disabling CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["DISABLE_CUDA"] = "1"
            os.environ["FORCE_CPU_ONLY"] = "1"
            return False
        
        # Check CUDA availability
        cuda_available = check_cuda_availability()
        
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
            
            # Add potential CUDA paths
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                cuda_bin = os.path.join(cuda_path, "bin")
                if os.path.exists(cuda_bin):
                    potential_paths.append(cuda_bin)
            
            # Try known CUDA paths on Windows
            if platform.system() == "Windows":
                for cuda_ver in ["v12.0", "v12.1", "v12.2", "v12.3", "v12.4", "v12.5", "v12.6", "v12.7", "v12.8", "v12.9"]:
                    cuda_dir = os.path.join("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA", cuda_ver, "bin")
                    if os.path.exists(cuda_dir):
                        potential_paths.append(cuda_dir)
            
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
        
        # Log additional debug information
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Current PATH: {os.environ.get('PATH', 'Not available')}")
        # Log all files in the llama_cpp/lib directory if it exists
        try:
            bundle_dir = getattr(sys, '_MEIPASS', None) or os.path.dirname(__file__)
            lib_dir = os.path.join(bundle_dir, 'llama_cpp', 'lib')
            if os.path.exists(lib_dir):
                logger.info(f"Files in {lib_dir}: {os.listdir(lib_dir)}")
            else:
                logger.warning(f"llama_cpp/lib directory does not exist at: {lib_dir}")
        except Exception as e:
            logger.warning(f"Error listing files in lib directory: {e}")
        # Log ctypes.util.find_library results
        for libname in ["llama", "ggml", "cublas", "cudart"]:
            found = ctypes.util.find_library(libname)
            logger.info(f"ctypes.util.find_library('{libname}') = {found}")
        
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
        
        # Log module and library details
        logger.info(f"llama_cpp module location: {llama_cpp.__file__}")
        logger.info(f"llama_cpp version: {getattr(llama_cpp, '__version__', 'unknown')}")
        
        # Check if CUDA is enabled in this build
        has_cuda = False
        try:
            if hasattr(llama_cpp, "llama") and hasattr(llama_cpp.llama, "Llama"):
                # Try creating a small model instance with 1 GPU layer to check if CUDA works
                logger.info("Testing CUDA support in llama_cpp...")
                
                # Instead of testing with a real model, we'll check if CUDA-related functions are available
                cuda_funcs = []
                
                # Get the library and check for CUDA functions
                if hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib"):
                    lib = llama_cpp.llama_cpp._lib
                    # Check for common CUDA function patterns
                    for attr_name in dir(lib):
                        if "cuda" in attr_name.lower() or "gpu" in attr_name.lower():
                            cuda_funcs.append(attr_name)
                
                if cuda_funcs:
                    logger.info(f"Found {len(cuda_funcs)} CUDA-related functions: {', '.join(cuda_funcs[:5])}")
                    has_cuda = True
                else:
                    logger.info("No CUDA functions found in llama_cpp library")
        except Exception as cuda_check_err:
            logger.warning(f"Error checking CUDA support: {cuda_check_err}")
            
        # Log CUDA support status
        logger.info(f"CUDA support detected in llama-cpp-python: {has_cuda}")
        
        # Patch the library initialization
        logger.info("Initializing llama-cpp backend...")
        
        # Find the library in either direct _lib or nested module structure
        lib = None
        
        # First check if llama_backend_init already exists and works
        if hasattr(llama_cpp, 'llama_backend_init'):
            logger.info("llama_backend_init function already exists, using directly")
            try:
                # Try calling the existing function
                logger.info("Trying existing llama_backend_init...")
                
                # Check if it takes parameters
                import inspect
                sig = inspect.signature(llama_cpp.llama_backend_init)
                params = list(sig.parameters.keys())
                
                if params:
                    logger.info(f"llama_backend_init takes parameters: {params}")
                else:
                    logger.info("llama_backend_init takes no parameters")
                
                # Call without parameters - this works for newer versions
                llama_cpp.llama_backend_init()
                logger.info("Successfully called existing llama_backend_init")
                return True
            except Exception as e:
                logger.warning(f"Failed to use existing llama_backend_init: {e}")
                # Fall through to try alternate approaches
        
        # Try to find the C library
        try:
            lib = None
            # Check for nested module structure first
            if hasattr(llama_cpp, 'llama_cpp'):
                nested = getattr(llama_cpp, 'llama_cpp')
                # Check if _lib exists in the nested module
                if hasattr(nested, '_lib'):
                    lib = getattr(nested, '_lib')
                    logger.info("Found lib in llama_cpp.llama_cpp._lib")
            
            # Check direct module if nested approach failed
            if lib is None and hasattr(llama_cpp, '_lib'):
                lib = getattr(llama_cpp, '_lib')
                logger.info("Found lib in llama_cpp._lib")
            
            # Last resort - try direct import
            if lib is None:
                try:
                    nested_module = importlib.import_module("llama_cpp.llama_cpp")
                    if hasattr(nested_module, '_lib'):
                        lib = getattr(nested_module, '_lib')
                        logger.info("Imported llama_cpp.llama_cpp._lib directly")
                except Exception as import_err:
                    logger.warning(f"Could not find _lib in any known location: {str(import_err)}")
                    logger.info("Trying to continue without patching llama_backend_init")
                    return True  # Return True to continue without patching
            
            # If we couldn't find lib in any location
            if lib is None:
                logger.warning("Could not find _lib attribute in any known location")
                logger.info("Trying to continue without patching llama_backend_init")
                return True
                
            # Check if backend_init exists in lib
            if hasattr(lib, 'llama_backend_init'):
                logger.info("Found llama_backend_init in lib")
                
                # Call the C function directly - this often works when other approaches fail
                try:
                    logger.info("Calling C function directly...")
                    lib.llama_backend_init()
                    logger.info("Successfully called lib.llama_backend_init directly")
                    return True
                except Exception as direct_err:
                    logger.warning(f"Error calling lib.llama_backend_init directly: {direct_err}")
                    # Fall through to try patching
                
                # Only try to patch if we haven't succeeded yet
                logger.info("Attempting to patch llama_backend_init...")
                
                # Define a new function that calls the C function without arguments
                def patched_backend_init():
                    logger.info("Called patched backend init (no args)")
                    return lib.llama_backend_init()
                
                # Replace the function
                llama_cpp.llama_backend_init = patched_backend_init
                
                # Test the patched function
                try:
                    logger.info("Testing patched function...")
                    llama_cpp.llama_backend_init()
                    logger.info("Successfully called patched llama_backend_init")
                    return True
                except Exception as patch_err:
                    logger.error(f"Failed to call patched function: {patch_err}")
                    # Continue and try CPU-only as fallback
            else:
                logger.warning("llama_backend_init not found in lib")
        except Exception as lib_err:
            logger.error(f"Error accessing library: {lib_err}")
        
        # If we got here, we failed all attempts
        # If we reach here, backend initialization failed - let's try with CPU-only mode
        logger.warning("Backend initialization failed, trying CPU-only mode")
        # Force CPU-only mode
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["DISABLE_CUDA"] = "1"
        
        # Try initialization again with CPU-only
        logger.info("Attempting to initialize llama_backend_init in CPU-only mode...")
        try:
            if hasattr(llama_cpp, 'llama_backend_init'):
                llama_cpp.llama_backend_init()
                logger.info("CPU-only initialization successful")
                return True
        except Exception as cpu_err:
            logger.error(f"CPU-only initialization also failed: {str(cpu_err)}")
        
        # If we get here, all attempts have failed
        logger.error("All initialization attempts failed")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize llama-cpp: {str(e)}", exc_info=True)
        return False
