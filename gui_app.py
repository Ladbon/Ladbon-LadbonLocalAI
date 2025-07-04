import os
import sys

# Add parent directory to path to ensure the relative import works
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import and setup onefile loader first - this handles PyInstaller temp directory issues
try:
    from utils.onefile_loader import prepare_temp_directory
    # Already ran during import, but we can explicitly call it here if needed
    # prepare_temp_directory()
except Exception as e:
    # We can't use logger here yet, so use simple print
    print(f"Warning: Failed to initialize onefile loader: {e}")

# Load DLLs for llama-cpp-python before anything else
try:
    # Check for CPU-only mode environment variable
    force_cpu = os.environ.get('FORCE_CPU_ONLY') == '1'
    if force_cpu:
        print("CPU-only mode requested by environment variable")
    
    # Initialize with appropriate mode
    from utils.llamacpp_loader import setup_llamacpp_environment
    cuda_success = setup_llamacpp_environment(force_cpu=force_cpu)
    
    if cuda_success:
        print("Successfully initialized llama-cpp with CUDA support")
    else:
        print("Using CPU-only mode for llama-cpp (no CUDA)")
except Exception as dll_error:
    print(f"Warning: Failed to initialize llama-cpp environment: {dll_error}")

# Import and apply NumPy fixes before any other imports
# This needs to be at the very top before anything else runs
try:
    # Import and apply numpy fixes
    from utils.numpy_init_fix import apply_numpy_fixes
    apply_numpy_fixes()
except Exception as early_import_error:
    # We can't use logger here yet, so use simple print
    print(f"Warning: Failed to apply early NumPy fixes: {early_import_error}")

# Now continue with regular imports
import logging
import traceback
import ctypes

# Import our data paths utility
try:
    from utils.data_paths import get_logs_dir, first_run_migration

    # Check for first run and migrate data if needed
    first_run_migration()
    
    # Get the logs directory
    log_dir = get_logs_dir()
except Exception as e:
    # If the import fails, fall back to local logs directory
    print(f"Warning: Failed to import data_paths utility: {e}")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# Ensure logs directory exists
os.makedirs(log_dir, exist_ok=True)

# Create a timestamp for the log file
from datetime import datetime
log_file = os.path.join(log_dir, f'gui_bootstrap_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')

# Set up initial logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level for maximum verbosity during startup
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # Use UTF-8 encoding to handle special characters
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('gui_app')

logger.info(f"Starting application, Python {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Executable path: {sys.executable}")
logger.info(f"_MEIPASS (PyInstaller bundle): {getattr(sys, '_MEIPASS', 'Not running in PyInstaller bundle')}")

# Setup llama-cpp environment before importing it
try:
    logger.info("Setting up llama-cpp environment...")
    
    # Add DLL directory from package root if running as PyInstaller bundle
    if hasattr(sys, '_MEIPASS'):
        logger.info(f"Running from PyInstaller bundle at: {sys._MEIPASS}")  # type: ignore
        
        # For standalone exe, ensure the directory structure exists
        # This is needed because the onefile mode extracts to a temp directory
        meipass_dir = sys._MEIPASS  # type: ignore
        llama_cpp_lib_dir = os.path.join(meipass_dir, 'llama_cpp', 'lib')
        
        # Create the directory structure if it doesn't exist
        if not os.path.exists(llama_cpp_lib_dir):
            logger.info(f"Creating llama_cpp lib directory at: {llama_cpp_lib_dir}")
            try:
                os.makedirs(llama_cpp_lib_dir, exist_ok=True)
                
                # Look for DLLs in the root directory and copy them to the lib directory
                dll_files = [f for f in os.listdir(meipass_dir) if f.lower().endswith('.dll')]
                for dll in dll_files:
                    if any(dll_name in dll.lower() for dll_name in ['ggml', 'llama', 'llava']):
                        src_path = os.path.join(meipass_dir, dll)
                        dst_path = os.path.join(llama_cpp_lib_dir, dll)
                        logger.info(f"Copying {dll} to lib directory")
                        try:
                            import shutil
                            shutil.copy2(src_path, dst_path)
                        except Exception as copy_err:
                            logger.warning(f"Error copying {dll}: {copy_err}")
            except Exception as dir_err:
                logger.error(f"Error creating lib directory: {dir_err}")
        
        # Manually add DLL search paths for various possible locations
        try:
            # On Windows, we can use AddDllDirectory to permanently add search paths
            if os.name == 'nt':
                logger.info("Using Windows-specific DLL directory functions")
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                
                # Try each possible location
                possible_lib_paths = [
                    os.path.join(sys._MEIPASS, 'llama_cpp', 'lib'),  # type: ignore
                    os.path.join(sys._MEIPASS, 'lib'),  # type: ignore
                    os.path.join(sys._MEIPASS)  # type: ignore
                ]
                
                for lib_path in possible_lib_paths:
                    if os.path.exists(lib_path):
                        logger.info(f"Adding DLL directory: {lib_path}")
                        try:
                            # Convert to wide string for Windows API
                            lib_path_c = ctypes.c_wchar_p(lib_path)
                            # Add the directory to the DLL search path
                            handle = kernel32.AddDllDirectory(lib_path_c)
                            if handle:
                                logger.info(f"Successfully added DLL directory: {lib_path}")
                            else:
                                error = ctypes.get_last_error()
                                logger.warning(f"Failed to add DLL directory: {lib_path}, error code: {error}")
                        except Exception as e:
                            logger.warning(f"Error adding DLL directory: {lib_path}, error: {str(e)}")
                
                # Also add to PATH environment variable as a fallback
                for lib_path in possible_lib_paths:
                    if os.path.exists(lib_path):
                        logger.info(f"Adding to PATH environment: {lib_path}")
                        os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
                        
                        # List files in directory
                        try:
                            files = os.listdir(lib_path)
                            logger.info(f"Files in {lib_path}: {', '.join(files)}")
                        except Exception as e:
                            logger.warning(f"Error listing files in {lib_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error setting up DLL directories: {str(e)}")
            
    logger.info("Setting up llama-cpp environment...")
    
    # If running as PyInstaller bundle, adjust DLL search path
    if getattr(sys, '_MEIPASS', None):
        logger.info("Running as PyInstaller bundle, configuring DLL search paths")
        bundle_dir = sys._MEIPASS  # type: ignore # PyInstaller adds this at runtime
        
        # Add potential DLL locations to PATH
        potential_lib_dirs = [
            os.path.join(bundle_dir, 'llama_cpp', 'lib'),
            os.path.join(bundle_dir, 'lib'),
            bundle_dir
        ]
        
        for lib_dir in potential_lib_dirs:
            if os.path.exists(lib_dir):
                logger.info(f"Adding {lib_dir} to PATH")
                os.environ['PATH'] = lib_dir + os.pathsep + os.environ.get('PATH', '')
                
                # List files in directory for debugging
                try:
                    files = os.listdir(lib_dir)
                    logger.info(f"Files in {lib_dir}: {', '.join(files)}")
                except Exception as e:
                    logger.error(f"Error listing files in {lib_dir}: {e}")
    
    # Now import and use the loader
    from utils.llamacpp_loader import setup_llamacpp_environment, initialize_llamacpp
    
    # Setup the environment first (paths, etc)
    env_setup = setup_llamacpp_environment()
    logger.info(f"Environment setup result: {'Success' if env_setup else 'Failed'}")
    
    # Then initialize the library
    lib_init = initialize_llamacpp()
    logger.info(f"Library initialization result: {'Success' if lib_init else 'Failed'}")
    
    logger.info("llama-cpp setup complete")
except Exception as e:
    logger.error(f"Failed to setup llama-cpp environment: {str(e)}")
    logger.error("Error details:\n" + traceback.format_exc())

# Import PyQt and other modules only after environment setup
from PyQt5.QtWidgets import QApplication, QMessageBox
from api.app import LocalAIApp

def main():
    # Initialize the application
    app = QApplication(sys.argv)
    
    try:
        logger.info("Creating LocalAIApp window...")
        window = LocalAIApp()
        window.show()
        logger.info("Application window shown successfully")
        return app.exec_()
    except Exception as e:
        logger.critical(f"Fatal error in application: {str(e)}", exc_info=True)
        
        # Check for the NumPy CPU dispatcher error specifically
        error_str = str(e)
        if "CPU dispatcher tracer already initialized" in error_str:
            # Special handling for NumPy dispatcher error
            error_msg = (
                "NumPy CPU dispatcher initialization error detected.\n\n"
                "This is likely caused by NumPy being imported multiple times through different paths.\n\n"
                "The application has created debug logs in the application directory.\n"
                "Please check these logs for more information."
            )
            QMessageBox.critical(None, "NumPy Initialization Error", error_msg)
        else:
            # Generic error handling
            error_msg = f"Error starting application:\n\n{error_str}"
            QMessageBox.critical(None, "Application Error", error_msg)
        
        return 1

if __name__ == "__main__":
    # First run the environment check
    try:
        from utils.check_environment import verify_environment
        env_results = verify_environment()
        
        # Log environment information
        logger.info(f"Python version: {env_results['python_version']}")
        logger.info(f"Running as: {'64-bit' if env_results['is_64bit_python'] else '32-bit'} Python")
        
        # Check for critical issues
        critical_issues = [issue for issue in env_results.get('issues', []) if "CRITICAL" in issue]
        if critical_issues:
            logger.error("CRITICAL ENVIRONMENT ISSUES DETECTED:")
            for issue in critical_issues:
                logger.error(f" - {issue}")
                
            # Show error dialog if running in GUI mode
            if not sys.stdout.isatty():  # Check if running with GUI (not in terminal)
                from PyQt5.QtWidgets import QMessageBox, QApplication
                app = QApplication(sys.argv)
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Environment Error")
                msg.setText("Critical issues detected with your environment")
                msg.setInformativeText("\n".join(critical_issues))
                msg.setDetailedText("Please reinstall the application using the latest installer, "
                                    "which will ensure it runs in 64-bit mode. "
                                    "If issues persist, contact support.")
                msg.exec_()
                sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to run environment check: {e}")
    
    logger.info("Starting main application (gui_app.py)...") 
    sys.exit(main())