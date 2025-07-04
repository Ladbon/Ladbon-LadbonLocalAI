"""
This script tests the CUDA initialization in the packaged app environment.
It should be copied to the packaged app directory and run from there.
"""

import os
import sys
import logging
import traceback
import ctypes
from ctypes import WinDLL
import glob

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cuda_test.log", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("cuda_test")

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

def check_cuda_dlls():
    """Check for CUDA DLLs in the environment"""
    logger.info("Checking for CUDA DLLs...")
    
    # Check PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    logger.info(f"PATH has {len(path_dirs)} directories")
    
    cuda_dlls = []
    for path in path_dirs:
        if not os.path.exists(path):
            continue
            
        # Look for CUDA DLLs
        for dll_name in ["cudart64_12.dll", "cudart64_120.dll", "cublas64_12.dll"]:
            dll_path = os.path.join(path, dll_name)
            if os.path.exists(dll_path):
                cuda_dlls.append(dll_path)
                logger.info(f"Found CUDA DLL in PATH: {dll_path}")
    
    if not cuda_dlls:
        logger.warning("No CUDA DLLs found in PATH")
    
    return cuda_dlls

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

def test_cuda_initialization():
    """Test the CUDA initialization process"""
    logger.info("===== TESTING CUDA INITIALIZATION =====")
    
    # Current directory
    app_dir = os.getcwd()
    logger.info(f"Current directory: {app_dir}")
    
    # If running from PyInstaller bundle
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        logger.info(f"Running from PyInstaller bundle: {meipass}")
        app_dir = meipass
    
    # Check for llama_cpp directory
    llama_cpp_dir = None
    possible_locations = [
        os.path.join(app_dir, "_internal", "llama_cpp"),
        os.path.join(app_dir, "llama_cpp"),
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            llama_cpp_dir = loc
            logger.info(f"Found llama_cpp directory at: {llama_cpp_dir}")
            break
    
    if not llama_cpp_dir:
        logger.error("Could not find llama_cpp directory!")
        return False
        
    # Check for lib directory
    lib_dir = os.path.join(llama_cpp_dir, "lib")
    if not os.path.exists(lib_dir):
        logger.error(f"No lib directory found at {lib_dir}")
        return False
        
    logger.info(f"Found llama_cpp/lib directory at: {lib_dir}")
    
    # List DLLs in the lib directory
    lib_dlls = glob.glob(os.path.join(lib_dir, "*.dll"))
    logger.info(f"Found {len(lib_dlls)} DLLs in lib directory:")
    for dll in lib_dlls:
        logger.info(f"  {os.path.basename(dll)}")
    
    # Check for CUDA DLLs
    cuda_dlls = [dll for dll in lib_dlls if "cuda" in os.path.basename(dll).lower()]
    if cuda_dlls:
        logger.info(f"Found {len(cuda_dlls)} CUDA DLLs in lib directory")
    else:
        logger.warning("No CUDA DLLs found in lib directory")
    
    # Add lib directory to DLL search path
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib_dir)
            logger.info(f"Added {lib_dir} to DLL search path")
        else:
            os.environ["PATH"] = lib_dir + os.pathsep + os.environ["PATH"]
            logger.info(f"Added {lib_dir} to PATH environment variable")
    except Exception as e:
        logger.error(f"Failed to add {lib_dir} to DLL search path: {e}")
    
    # Try to import llama_cpp
    try:
        logger.info("Attempting to import llama_cpp...")
        import llama_cpp
        logger.info(f"Successfully imported llama_cpp from {llama_cpp.__file__}")
        
        # Try to initialize the backend with simple approach
        success = False
        logger.info("Trying llama_backend_init()...")
        
        try:
            # Try the standard signature first
            llama_cpp.llama_backend_init()
            logger.info("Backend initialization succeeded!")
            success = True
        except TypeError as te:
            # This might be a signature error
            logger.error(f"Backend initialization failed with TypeError: {te}")
            logger.error("This might be due to a signature mismatch. Check llama-cpp-python version.")
            logger.error(traceback.format_exc())
        except Exception as e:
            # Other errors like access violation
            logger.error(f"Backend initialization failed: {e}")
            logger.error(traceback.format_exc())
        
        if not success:
            logger.error("All backend initialization methods failed")
            
            # Try CPU-only mode as last resort
            try:
                logger.info("Trying CPU-only mode...")
                os.environ["LLAMA_CPP_CPU_ONLY"] = "1"
                llama_cpp.llama_backend_init()
                logger.info("CPU-only backend initialization succeeded")
                return True
            except Exception as e:
                logger.error(f"CPU-only backend initialization failed: {e}")
                return False
        
        return success
    
    except ImportError as ie:
        logger.error(f"Failed to import llama_cpp: {ie}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_model_loading():
    """Try to load a small test model"""
    logger.info("===== TESTING MODEL LOADING =====")
    
    # First check if we have models directory
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found at {models_dir}")
        logger.warning("Skipping model loading test")
        return False
    
    # Find a small model to test with
    test_models = glob.glob(os.path.join(models_dir, "*.gguf"))
    if not test_models:
        logger.warning("No .gguf models found in models directory")
        logger.warning("Skipping model loading test")
        return False
    
    # Sort by file size and pick the smallest
    test_models.sort(key=os.path.getsize)
    test_model = test_models[0]
    logger.info(f"Using {test_model} for testing")
    
    try:
        logger.info("Attempting to import llama_cpp...")
        import llama_cpp
        
        # Set CPU-only mode to test basic model loading
        os.environ["LLAMA_CPP_CPU_ONLY"] = "1"
        
        # Try to initialize the backend
        logger.info("Initializing llama_cpp backend in CPU mode...")
        try:
            llama_cpp.llama_backend_init()
            logger.info("Backend initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            return False
        
        # Try to load the model
        logger.info(f"Loading model {os.path.basename(test_model)}...")
        try:
            model = llama_cpp.Llama(
                model_path=test_model,
                n_ctx=512,  # Use small context for test
                n_batch=512  # Small batch for test
            )
            logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            return False
    
    except ImportError as ie:
        logger.error(f"Failed to import llama_cpp: {ie}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_cuda_model_loading():
    """Try to load a small test model with CUDA"""
    logger.info("===== TESTING CUDA MODEL LOADING =====")
    
    # First check if we have models directory
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found at {models_dir}")
        logger.warning("Skipping CUDA model loading test")
        return False
    
    # Find a small model to test with
    test_models = glob.glob(os.path.join(models_dir, "*.gguf"))
    if not test_models:
        logger.warning("No .gguf models found in models directory")
        logger.warning("Skipping CUDA model loading test")
        return False
    
    # Sort by file size and pick the smallest
    test_models.sort(key=os.path.getsize)
    test_model = test_models[0]
    logger.info(f"Using {test_model} for CUDA testing")
    
    try:
        logger.info("Attempting to import llama_cpp...")
        import llama_cpp
        
        # Make sure CPU-only mode is not set
        if "LLAMA_CPP_CPU_ONLY" in os.environ:
            del os.environ["LLAMA_CPP_CPU_ONLY"]
        
        # Try to initialize the backend with CUDA
        logger.info("Initializing llama_cpp backend with CUDA...")
        try:
            llama_cpp.llama_backend_init()
            logger.info("Backend initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize backend with CUDA: {e}")
            return False
        
        # Try to load the model with CUDA
        logger.info(f"Loading model with CUDA {os.path.basename(test_model)}...")
        try:
            model = llama_cpp.Llama(
                model_path=test_model,
                n_ctx=512,  # Use small context for test
                n_batch=512,  # Small batch for test
                n_gpu_layers=1  # Use at least 1 GPU layer to test CUDA
            )
            logger.info("Model loaded successfully with CUDA!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model with CUDA: {e}")
            logger.error(traceback.format_exc())
            return False
    
    except ImportError as ie:
        logger.error(f"Failed to import llama_cpp: {ie}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main test function"""
    logger.info("===== CUDA TEST SCRIPT =====")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Script location: {__file__}")
    
    # Test environment
    logger.info(f"PATH: {os.environ.get('PATH')}")
    
    # Check CUDA environment variables
    for env_var in ["CUDA_PATH", "CUDA_HOME", "LLAMA_CPP_CPU_ONLY"]:
        logger.info(f"{env_var}={os.environ.get(env_var, 'Not set')}")
    
    # List DLLs in current directory and _internal directory
    list_dlls_in_directory(os.getcwd())
    
    internal_dir = os.path.join(os.getcwd(), "_internal")
    if os.path.exists(internal_dir):
        list_dlls_in_directory(internal_dir)
    
    # Check for CUDA DLLs in PATH
    check_cuda_dlls()
    
    # Test CUDA initialization
    cuda_init_result = test_cuda_initialization()
    logger.info(f"CUDA initialization test: {'PASSED' if cuda_init_result else 'FAILED'}")
    
    # Test basic model loading (CPU)
    model_load_result = test_model_loading()
    logger.info(f"Model loading test (CPU): {'PASSED' if model_load_result else 'FAILED'}")
    
    # Test CUDA model loading
    cuda_model_load_result = test_cuda_model_loading()
    logger.info(f"Model loading test (CUDA): {'PASSED' if cuda_model_load_result else 'FAILED'}")
    
    # Summary
    logger.info("===== TEST SUMMARY =====")
    logger.info(f"CUDA initialization: {'PASSED' if cuda_init_result else 'FAILED'}")
    logger.info(f"Model loading (CPU): {'PASSED' if model_load_result else 'FAILED'}")
    logger.info(f"Model loading (CUDA): {'PASSED' if cuda_model_load_result else 'FAILED'}")
    
    if cuda_model_load_result:
        logger.info("SUCCESS: CUDA is working properly!")
    else:
        logger.warning("CUDA is not working properly. Check the logs for details.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
