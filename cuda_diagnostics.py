"""
CUDA Diagnostics Tool for llama-cpp-python

This script provides detailed diagnostics for CUDA availability and DLL loading issues.
It's designed to work both in development environment and in PyInstaller bundled apps.
"""

import os
import sys
import ctypes
import glob
import logging
import platform
import importlib
import traceback
from pathlib import Path
from ctypes.util import find_library

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "cuda_diagnostics.log"), mode='w')
    ]
)

logger = logging.getLogger("cuda_diagnostics")

def check_cuda_environment():
    """Check CUDA environment variables and paths"""
    logger.info("=== CUDA Environment Diagnostics ===")
    
    try:
        # Check CUDA environment variables
        cuda_env_vars = {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "CUDA_PATH": os.environ.get("CUDA_PATH"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        }
        
        for var, value in cuda_env_vars.items():
            logger.info(f"{var}: {value}")
        
        # Check PATH for CUDA directories
        paths = os.environ.get("PATH", "").split(os.pathsep)
    except Exception as e:
        logger.error(f"Error checking CUDA environment: {e}")
        paths = []
    cuda_paths = [p for p in paths if "cuda" in p.lower()]
    logger.info(f"CUDA directories in PATH: {len(cuda_paths)}")
    for path in cuda_paths:
        logger.info(f"  - {path}")
    
    # Check for common CUDA DLLs
    cuda_dlls = []
    cuda_dll_names = ["cudart64", "cublas64", "cublasLt64", "curand64"]
    
    for path in paths:
        if os.path.exists(path):
            for dll_name in cuda_dll_names:
                matches = list(Path(path).glob(f"{dll_name}*.dll"))
                for match in matches:
                    cuda_dlls.append(str(match))
    
    logger.info(f"Found {len(cuda_dlls)} CUDA DLLs in PATH:")
    for dll in cuda_dlls:
        logger.info(f"  - {dll}")
    
    return len(cuda_dlls) > 0

def find_and_list_cuda_dlls():
    """Find CUDA DLLs in common locations"""
    logger.info("=== Finding CUDA DLLs ===")
    
    cuda_paths = []
    
    # Check common CUDA installation paths
    if platform.system() == "Windows":
        base_dirs = [
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
            "C:/Program Files/NVIDIA Corporation"
        ]
        
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                # Look for version subdirectories
                for item in os.listdir(base_dir):
                    version_dir = os.path.join(base_dir, item)
                    if os.path.isdir(version_dir) and ("v" in item or "." in item):
                        bin_dir = os.path.join(version_dir, "bin")
                        if os.path.exists(bin_dir):
                            cuda_paths.append(bin_dir)
    
    # Look for DLLs in found CUDA paths
    all_dlls = []
    for path in cuda_paths:
        dlls = list(Path(path).glob("*.dll"))
        logger.info(f"Found {len(dlls)} DLLs in {path}")
        all_dlls.extend(dlls)
    
    # List critical CUDA DLLs
    critical_dlls = {
        "cudart": [],
        "cublas": [],
        "cublasLt": [],
        "curand": []
    }
    
    for dll in all_dlls:
        dll_name = dll.name.lower()
        for key in critical_dlls:
            if key in dll_name:
                critical_dlls[key].append(str(dll))
    
    for key, dlls in critical_dlls.items():
        logger.info(f"{key} DLLs found: {len(dlls)}")
        for dll in dlls:
            logger.info(f"  - {dll}")
    
    return critical_dlls

def check_llama_cpp_installation():
    """Check llama-cpp-python installation"""
    logger.info("=== Checking llama-cpp-python installation ===")
    
    try:
        import llama_cpp
        logger.info(f"llama_cpp module location: {llama_cpp.__file__}")
        logger.info(f"llama_cpp version: {getattr(llama_cpp, '__version__', 'unknown')}")
        
        # Check if we're in a PyInstaller bundle
        is_bundled = getattr(sys, 'frozen', False)
        if is_bundled:
            logger.info("Running from PyInstaller bundle")
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            logger.info(f"Bundle directory: {base_dir}")
            
            # Check DLL directories
            dll_dirs = [
                os.path.join(base_dir, "_internal/llama_cpp/lib"),
                os.path.join(base_dir, "llama_cpp/lib")
            ]
            
            for dll_dir in dll_dirs:
                if os.path.exists(dll_dir):
                    logger.info(f"DLL directory exists: {dll_dir}")
                    dlls = list(Path(dll_dir).glob("*.dll"))
                    logger.info(f"Found {len(dlls)} DLLs in {dll_dir}")
                    for dll in dlls:
                        logger.info(f"  - {dll.name}")
                else:
                    logger.warning(f"DLL directory does not exist: {dll_dir}")
        
        # Check if Llama class exists and what backends are available
        if hasattr(llama_cpp, "Llama"):
            logger.info("Llama class is available")
            
            # Check for GPU capabilities
            try:
                backend_info = {}
                if hasattr(llama_cpp, "llama_backend_init"):
                    backend_info["llama_backend_init"] = "Available directly in module"
                
                if hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib"):
                    lib = llama_cpp.llama_cpp._lib
                    cuda_funcs = []
                    for attr_name in dir(lib):
                        if "cuda" in attr_name.lower() or "gpu" in attr_name.lower():
                            cuda_funcs.append(attr_name)
                    
                    if cuda_funcs:
                        backend_info["cuda_functions"] = cuda_funcs
                
                if backend_info:
                    logger.info(f"GPU-related backend features: {backend_info}")
                else:
                    logger.warning("No GPU-related backend features found")
            
            except Exception as backend_err:
                logger.error(f"Error checking backend capabilities: {backend_err}")
        else:
            logger.warning("Llama class is not available")
        
        return True
    
    except ImportError as e:
        logger.error(f"Failed to import llama_cpp: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Error checking llama_cpp installation: {e}")
        logger.error(traceback.format_exc())
        return False

def try_cuda_initialization():
    """Try to initialize CUDA backend in llama-cpp-python"""
    logger.info("=== Attempting CUDA initialization ===")
    
    try:
        import llama_cpp
        
        # Find the right function and try to call it
        backend_init_func = None
        
        # Check different locations
        if hasattr(llama_cpp, "llama_backend_init"):
            backend_init_func = llama_cpp.llama_backend_init
            logger.info("Found llama_backend_init directly in module")
        elif hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib") and hasattr(llama_cpp.llama_cpp._lib, "llama_backend_init"):
            backend_init_func = llama_cpp.llama_cpp._lib.llama_backend_init
            logger.info("Found llama_backend_init in llama_cpp._lib")
        
        if backend_init_func:
            logger.info("Calling llama_backend_init...")
            
            try:
                # Try different ways of calling the function
                try:
                    # Try with numa=False parameter
                    import inspect
                    sig = inspect.signature(backend_init_func)
                    params = list(sig.parameters.keys())
                    
                    if len(params) > 0:
                        logger.info(f"Calling with parameters: {params}")
                        # Use a dynamic approach that doesn't trigger Pylance warnings
                        param_name = params[0]  # Get the first parameter name
                        try:
                            # Try using kwargs
                            kwargs = {param_name: False}
                            backend_init_func(**kwargs)
                        except Exception:
                            # If kwargs fails, try no parameters
                            backend_init_func()
                    else:
                        logger.info("Calling without parameters")
                        backend_init_func()
                
                except TypeError as type_err:
                    logger.warning(f"TypeError when calling backend_init: {type_err}")
                    # Try alternative calling method
                    if "takes 0 positional arguments" in str(type_err):
                        logger.info("Retrying without arguments")
                        backend_init_func()
                    elif "missing 1 required positional argument" in str(type_err) or "unexpected keyword argument" in str(type_err):
                        logger.info("Trying with C function directly")
                        # Try to access the underlying C function if available
                        try:
                            if hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib"):
                                logger.info("Calling C function directly via _lib")
                                llama_cpp.llama_cpp._lib.llama_backend_init()
                            else:
                                logger.info("No direct C function access available")
                        except Exception as e:
                            logger.error(f"Error calling C function: {e}")
                    else:
                        raise
                
                logger.info("Successfully initialized backend!")
                return True
            
            except Exception as init_err:
                logger.error(f"Error initializing backend: {init_err}")
                logger.error(traceback.format_exc())
                return False
        else:
            logger.error("Could not find llama_backend_init function")
            return False
    
    except Exception as e:
        logger.error(f"Error during CUDA initialization attempt: {e}")
        logger.error(traceback.format_exc())
        return False

def check_gpu_model_loading():
    """Try to load a small model with GPU enabled"""
    logger.info("=== Testing GPU model loading ===")
    
    try:
        import llama_cpp
        
        # Test GPU model loading with minimal configuration
        model_path = os.environ.get("LLAMA_CPP_TEST_MODEL")
        if not model_path:
            logger.warning("No test model specified in LLAMA_CPP_TEST_MODEL environment variable")
            logger.warning("Skipping model loading test")
            return False
        
        if not os.path.exists(model_path):
            logger.warning(f"Test model does not exist: {model_path}")
            logger.warning("Skipping model loading test")
            return False
        
        logger.info(f"Attempting to load model with GPU: {model_path}")
        
        # Try to load with 1 layer on GPU
        try:
            logger.info("Creating Llama model with n_gpu_layers=1...")
            model = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=1,
                verbose=True
            )
            logger.info("Successfully loaded model with GPU!")
            return True
        
        except Exception as model_err:
            logger.error(f"Error loading model with GPU: {model_err}")
            logger.error(traceback.format_exc())
            
            # Try with CPU as fallback
            try:
                logger.info("Trying with CPU fallback (n_gpu_layers=0)...")
                model = llama_cpp.Llama(
                    model_path=model_path,
                    n_gpu_layers=0,
                    verbose=True
                )
                logger.info("Successfully loaded model with CPU only!")
                return False  # Return False since GPU failed
            
            except Exception as cpu_err:
                logger.error(f"Error loading model with CPU: {cpu_err}")
                return False
    
    except ImportError as e:
        logger.error(f"Failed to import llama_cpp for model test: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error during model loading test: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_cuda_paths():
    """
    Try to fix CUDA paths by adding necessary directories to PATH
    Returns True if paths were modified
    """
    logger.info("=== Fixing CUDA paths ===")
    
    # Check if we're in a PyInstaller bundle
    is_bundled = getattr(sys, 'frozen', False)
    bundle_dir = getattr(sys, '_MEIPASS', None) if is_bundled else None
    
    if bundle_dir:
        logger.info(f"Running from PyInstaller bundle: {bundle_dir}")
        
        # Add potential DLL locations to PATH
        dll_dirs = [
            os.path.join(bundle_dir, "_internal", "llama_cpp", "lib"),
            os.path.join(bundle_dir, "llama_cpp", "lib"),
            os.path.join(bundle_dir, "_internal", "cuda_dlls"),
            os.path.join(bundle_dir, "cuda_dlls"),
        ]
        
        paths_added = False
        for dll_dir in dll_dirs:
            if os.path.exists(dll_dir):
                # Add the directory to DLL search path
                try:
                    os.add_dll_directory(dll_dir)
                    logger.info(f"Added DLL directory: {dll_dir}")
                    paths_added = True
                except Exception as e:
                    logger.error(f"Error adding DLL directory {dll_dir}: {e}")
                
                # Update PATH environment variable
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
                logger.info(f"Added to PATH: {dll_dir}")
                paths_added = True
        
        # Look for CUDA in system
        cuda_paths = []
        if platform.system() == "Windows":
            base_dirs = [
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
                "C:/Program Files/NVIDIA Corporation"
            ]
            
            for base_dir in base_dirs:
                if os.path.exists(base_dir):
                    # Look for version subdirectories
                    for item in os.listdir(base_dir):
                        version_dir = os.path.join(base_dir, item)
                        if os.path.isdir(version_dir) and ("v" in item or "." in item):
                            bin_dir = os.path.join(version_dir, "bin")
                            if os.path.exists(bin_dir):
                                cuda_paths.append(bin_dir)
        
        # Add CUDA directories to PATH
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                # Add the directory to DLL search path
                try:
                    os.add_dll_directory(cuda_path)
                    logger.info(f"Added CUDA DLL directory: {cuda_path}")
                    paths_added = True
                except Exception as e:
                    logger.error(f"Error adding CUDA DLL directory {cuda_path}: {e}")
                
                # Update PATH environment variable
                os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")
                logger.info(f"Added CUDA path to PATH: {cuda_path}")
                paths_added = True
        
        logger.info(f"Final PATH: {os.environ.get('PATH')}")
        return paths_added
    
    else:
        logger.info("Not running from PyInstaller bundle, no path fixing needed")
        return False

def recommend_fixes():
    """Recommend fixes based on diagnostic results"""
    logger.info("=== Recommendations ===")
    
    # Create a batch file for launching with correct CUDA paths
    is_bundled = getattr(sys, 'frozen', False)
    if is_bundled:
        executable = sys.executable
        base_dir = os.path.dirname(executable)
        
        cuda_paths = []
        if platform.system() == "Windows":
            base_dirs = [
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
                "C:/Program Files/NVIDIA Corporation"
            ]
            
            for base_dir in base_dirs:
                if os.path.exists(base_dir):
                    for item in os.listdir(base_dir):
                        version_dir = os.path.join(base_dir, item)
                        if os.path.isdir(version_dir) and ("v" in item or "." in item):
                            bin_dir = os.path.join(version_dir, "bin")
                            if os.path.exists(bin_dir):
                                cuda_paths.append(bin_dir)
        
        # Generate a launcher batch file
        launcher_path = os.path.join(base_dir, "Launch_with_CUDA.bat")
        with open(launcher_path, "w") as f:
            f.write("@echo off\n")
            f.write("echo Setting up CUDA environment for Ladbon AI Desktop...\n")
            f.write("\n")
            
            # Add CUDA paths to PATH
            for cuda_path in cuda_paths:
                f.write(f'set "PATH={cuda_path};%PATH%"\n')
            
            # Add DLL directories from bundle
            f.write(f'set "PATH=%~dp0_internal\\llama_cpp\\lib;%PATH%"\n')
            f.write(f'set "PATH=%~dp0_internal\\cuda_dlls;%PATH%"\n')
            
            f.write("\n")
            f.write("echo Launching application with CUDA support...\n")
            f.write(f'"%~dp0{os.path.basename(executable)}"\n')
        
        logger.info(f"Created launcher batch file: {launcher_path}")
        
        # Generate a diagnostics script that can be run to check CUDA setup
        diagnostics_path = os.path.join(base_dir, "Check_CUDA.bat")
        with open(diagnostics_path, "w") as f:
            f.write("@echo off\n")
            f.write("echo Checking CUDA setup...\n")
            f.write("\n")
            
            # Add CUDA paths to PATH
            for cuda_path in cuda_paths:
                f.write(f'set "PATH={cuda_path};%PATH%"\n')
            
            # Add DLL directories from bundle
            f.write(f'set "PATH=%~dp0_internal\\llama_cpp\\lib;%PATH%"\n')
            f.write(f'set "PATH=%~dp0_internal\\cuda_dlls;%PATH%"\n')
            
            f.write("\n")
            f.write('echo Current PATH:\n')
            f.write('echo %PATH%\n')
            f.write("\n")
            f.write('echo Checking for CUDA DLLs:\n')
            f.write('where cudart64*.dll\n')
            f.write('where cublas64*.dll\n')
            f.write("\n")
            f.write('echo Running CUDA diagnostics...\n')
            f.write(f'"%~dp0_internal\\python.exe" "%~dp0_internal\\cuda_diagnostics.py"\n')
            f.write("\n")
            f.write("pause\n")
        
        logger.info(f"Created diagnostics batch file: {diagnostics_path}")
    
    logger.info("Recommendations:")
    logger.info("1. Make sure CUDA DLLs are properly installed on your system")
    logger.info("2. Use the Launch_with_CUDA.bat script to launch the application")
    logger.info("3. Check the logs directory for detailed diagnostic information")

if __name__ == "__main__":
    logger.info("Starting CUDA Diagnostics Tool")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Step 1: Fix CUDA paths
    fix_cuda_paths()
    
    # Step 2: Check CUDA environment
    has_cuda_env = check_cuda_environment()
    
    # Step 3: Find CUDA DLLs
    cuda_dlls = find_and_list_cuda_dlls()
    
    # Step 4: Check llama-cpp-python installation
    llama_cpp_ok = check_llama_cpp_installation()
    
    # Step 5: Try CUDA initialization
    cuda_init_ok = try_cuda_initialization() if llama_cpp_ok else False
    
    # Step 6: Recommend fixes
    recommend_fixes()
    
    # Final summary
    logger.info("\n=== Diagnostics Summary ===")
    logger.info(f"CUDA environment available: {has_cuda_env}")
    logger.info(f"llama-cpp-python installation OK: {llama_cpp_ok}")
    logger.info(f"CUDA initialization successful: {cuda_init_ok}")
    
    if cuda_init_ok:
        logger.info("✅ CUDA is working properly!")
    else:
        logger.info("❌ CUDA initialization failed. Check the logs for details.")
