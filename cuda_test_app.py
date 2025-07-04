"""
CUDA Test for Packaged App
This script tests CUDA support in the packaged app by:
1. Checking PATH for CUDA DLLs
2. Directly loading CUDA DLLs using ctypes
3. Importing llama_cpp and initializing CUDA backend
4. Testing model loading with a small test model (if available)
"""

import os
import sys
import ctypes
import logging
import platform
from pathlib import Path
from datetime import datetime

# Set up logging
os.makedirs("logs", exist_ok=True)
log_file = f"logs/cuda_test_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("=" * 70)
logging.info("CUDA Test for Packaged App")
logging.info("=" * 70)

# System information
logging.info(f"Python: {platform.python_version()}")
logging.info(f"Platform: {platform.platform()}")
logging.info(f"Executable: {sys.executable}")
logging.info(f"Working directory: {os.getcwd()}")

# Get application directory
app_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
logging.info(f"Application directory: {app_dir}")

# Check PATH environment variable
logging.info("\n--- PATH Environment ---")
path_dirs = os.environ.get("PATH", "").split(os.pathsep)
for idx, directory in enumerate(path_dirs):
    logging.info(f"{idx+1}. {directory}")

# Search for CUDA DLLs in PATH
logging.info("\n--- CUDA DLLs in PATH ---")
cuda_dll_found = False
for directory in path_dirs:
    if not os.path.exists(directory):
        continue
    
    try:
        dlls = [f for f in os.listdir(directory) if f.lower().endswith(".dll") and 
                ("cuda" in f.lower() or "cublas" in f.lower())]
        
        if dlls:
            cuda_dll_found = True
            logging.info(f"Found in {directory}:")
            for dll in dlls:
                logging.info(f"  - {dll}")
    except Exception as e:
        logging.warning(f"Error listing directory {directory}: {e}")

if not cuda_dll_found:
    logging.warning("No CUDA DLLs found in PATH!")

# Create cuda_dlls directory if not exists
cuda_dlls_dir = os.path.join(app_dir, "cuda_dlls")
if not os.path.exists(cuda_dlls_dir):
    os.makedirs(cuda_dlls_dir, exist_ok=True)
    logging.info(f"Created {cuda_dlls_dir}")

# Test loading CUDA DLLs directly
logging.info("\n--- Direct CUDA DLL Loading Test ---")
dll_names = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
dll_results = {}

for dll_name in dll_names:
    try:
        lib = ctypes.CDLL(dll_name)
        logging.info(f"✓ Successfully loaded {dll_name}")
        dll_results[dll_name] = True
    except Exception as e:
        logging.error(f"✗ Failed to load {dll_name}: {e}")
        dll_results[dll_name] = False

# Test importing llama_cpp
logging.info("\n--- llama_cpp Import Test ---")
try:
    import llama_cpp
    logging.info("✓ Successfully imported llama_cpp")
    
    # Check version
    if hasattr(llama_cpp, "__version__"):
        logging.info(f"  Version: {llama_cpp.__version__}")
    else:
        logging.info("  Version: unknown")
    
    # Check if llama_cpp has required backend_init function
    if hasattr(llama_cpp, "llama_backend_init"):
        logging.info("✓ Found llama_backend_init function")
        backend_init_fn = llama_cpp.llama_backend_init
    elif hasattr(llama_cpp, "_lib") and hasattr(llama_cpp._lib, "llama_backend_init"):
        logging.info("✓ Found _lib.llama_backend_init function")
        backend_init_fn = llama_cpp._lib.llama_backend_init
    else:
        logging.error("✗ Could not find llama_backend_init function!")
        backend_init_fn = None
    
    # Try initializing CUDA backend
    if backend_init_fn:
        logging.info("\n--- CUDA Backend Initialization Test ---")
        try:
            # Try with argument (newer versions)
            try:
                logging.info("Attempting llama_backend_init(True)...")
                llama_cpp.llama_backend_init(True)
                logging.info("✓ Successfully initialized CUDA backend with llama_backend_init(True)")
            except (TypeError, AttributeError) as e:
                logging.warning(f"Method 1 failed: {e}")
                
                # Try with keyword argument
                try:
                    logging.info("Attempting llama_backend_init(use_cuda=True)...")
                    llama_cpp.llama_backend_init(use_cuda=True)
                    logging.info("✓ Successfully initialized CUDA backend with llama_backend_init(use_cuda=True)")
                except (TypeError, AttributeError) as e:
                    logging.warning(f"Method 2 failed: {e}")
                    
                    # Try direct _lib call
                    logging.info("Attempting _lib.llama_backend_init()...")
                    if hasattr(llama_cpp, "_lib") and hasattr(llama_cpp._lib, "llama_backend_init"):
                        llama_cpp._lib.llama_backend_init()
                        logging.info("✓ Successfully initialized CUDA backend with _lib.llama_backend_init()")
                    else:
                        logging.error("✗ _lib.llama_backend_init not available")
                        
        except Exception as e:
            logging.error(f"✗ Failed to initialize CUDA backend: {e}")
    
    # Try getting CUDA device count
    logging.info("\n--- CUDA Device Information ---")
    try:
        if hasattr(llama_cpp, "get_num_threads"):
            threads = llama_cpp.get_num_threads()
            logging.info(f"Number of threads: {threads}")
        
        # Check GPU device info using a Llama model
        try:
            from llama_cpp import Llama
            
            # Try to get system info without loading a model
            logging.info("Getting system information...")
            Llama.system_info()
            
            # Look for a small test model
            models_dir = os.path.join(os.path.dirname(app_dir), "models")
            test_model = None
            
            if os.path.exists(models_dir):
                logging.info(f"Checking for models in {models_dir}")
                model_files = []
                for root, _, files in os.walk(models_dir):
                    for file in files:
                        if file.endswith(('.gguf', '.bin')) and os.path.getsize(os.path.join(root, file)) < 500*1024*1024:  # < 500MB
                            model_files.append(os.path.join(root, file))
                
                if model_files:
                    test_model = min(model_files, key=lambda x: os.path.getsize(x))
                    logging.info(f"Found test model: {test_model} ({os.path.getsize(test_model) / 1024 / 1024:.2f} MB)")
            
            if test_model:
                logging.info(f"Trying to load model: {test_model}")
                model = Llama(model_path=test_model, n_gpu_layers=1, n_ctx=512, verbose=True)
                logging.info(f"✓ Successfully loaded model with GPU acceleration")
                
                # Generate a small test
                logging.info("Testing inference...")
                output = model("Once upon a time", max_tokens=5)
                logging.info(f"Model output: {output}")
        except Exception as e:
            logging.error(f"✗ Failed to check GPU or load model: {e}")
    except Exception as e:
        logging.error(f"✗ Error getting CUDA information: {e}")
        
except Exception as e:
    logging.error(f"✗ Failed to import llama_cpp: {e}")

# Summary
logging.info("\n" + "=" * 30 + " SUMMARY " + "=" * 30)

if all(dll_results.values()):
    logging.info("✓ All CUDA DLLs loaded successfully")
else:
    failed_dlls = [name for name, success in dll_results.items() if not success]
    logging.warning(f"✗ Failed to load some CUDA DLLs: {', '.join(failed_dlls)}")

if not cuda_dll_found:
    logging.warning("✗ No CUDA DLLs found in PATH!")

logging.info("=" * 70)
logging.info(f"Log file: {os.path.abspath(log_file)}")
logging.info("=" * 70)

print(f"\nTest completed! Log file: {os.path.abspath(log_file)}")
