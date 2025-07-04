# Ensures the CUDA environment is set up before any imports
import os
import sys
import glob
import logging

# Create global logger
def setup_logging():
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            filename=os.path.join(log_dir, "cuda_setup.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        return logging.getLogger("cuda_setup")
    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Return a basic logger if we fail
        return logging.getLogger("cuda_setup")

logger = setup_logging()

def setup_cuda_paths():
    """
    Add potential CUDA paths to the system PATH to help find DLLs
    This runs before any imports so the DLLs can be found when needed
    """
    try:
        logger.info("Setting up CUDA paths")
        
        # Get the bundle dir if running as PyInstaller package
        bundle_dir = getattr(sys, '_MEIPASS', None)
        if bundle_dir:
            logger.info(f"Running from PyInstaller bundle: {bundle_dir}")
        
        # Check known CUDA paths
        cuda_paths = []
        for ver in ["12.0", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.7", "12.8", "12.9"]:
            cuda_path = f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{ver}\\bin"
            if os.path.exists(cuda_path):
                cuda_paths.append(cuda_path)
                logger.info(f"Found CUDA {ver} bin directory: {cuda_path}")
        
        if cuda_paths:
            for path in cuda_paths:
                # Add to PATH
                if path not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                    logger.info(f"Added to PATH: {path}")
                    
                    # List CUDA DLLs in this directory
                    try:
                        cuda_dlls = glob.glob(os.path.join(path, "*.dll"))
                        logger.info(f"Found {len(cuda_dlls)} DLLs in {path}")
                        for i, dll in enumerate(cuda_dlls[:10]):
                            logger.info(f"  - {os.path.basename(dll)}")
                        if len(cuda_dlls) > 10:
                            logger.info(f"  - ... and {len(cuda_dlls) - 10} more")
                    except Exception as e:
                        logger.error(f"Error listing DLLs in {path}: {e}")
        else:
            logger.warning("No CUDA installations found")
            
        # Also add llama_cpp lib directory to path if in bundle
        if bundle_dir:
            llama_cpp_lib = os.path.join(bundle_dir, "llama_cpp", "lib")
            if os.path.exists(llama_cpp_lib):
                os.environ["PATH"] = llama_cpp_lib + os.pathsep + os.environ.get("PATH", "")
                logger.info(f"Added to PATH: {llama_cpp_lib}")
                
        logger.info(f"Final PATH: {os.environ.get('PATH', '')}")
        return True
    except Exception as e:
        try:
            logger.error(f"Error setting up CUDA paths: {e}", exc_info=True)
        except:
            print(f"Error setting up CUDA paths: {e}")
        return False

# Run setup on import
setup_cuda_paths()