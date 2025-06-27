"""
This module provides a fix for the NumPy CPU dispatcher initialization issue
when running in a PyInstaller bundle.
"""
import os
import sys
import logging
import traceback
from datetime import datetime

# Set up basic console logging first, since the regular logger might not be set up yet
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("numpy_init_fix")

# Create a debug log file directly in the current working directory
debug_log_path = os.path.join(os.getcwd(), f'numpy_fix_debug_{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
try:
    with open(debug_log_path, 'w') as f:
        f.write(f"NumPy fix debug log created at {datetime.now()}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Working directory: {os.getcwd()}\n")
        f.write(f"Is PyInstaller bundle: {hasattr(sys, '_MEIPASS')}\n")
        if hasattr(sys, '_MEIPASS'):
            f.write(f"PyInstaller _MEIPASS: {sys._MEIPASS}\n")  # type: ignore # PyInstaller adds this at runtime
except Exception as e:
    print(f"Error creating debug log: {e}")
    pass  # Don't crash if we can't create the log

def apply_numpy_fixes():
    """
    Apply fixes to prevent NumPy CPU dispatcher initialization issues.
    
    Call this function before any module that might import NumPy is imported.
    This should be the first thing called in the main script, even before
    setting up logging.
    """
    try:
        # Write to debug log
        with open(debug_log_path, 'a') as f:
            f.write("\nEntering apply_numpy_fixes()\n")
        
        # Set environment variables to restrict thread usage
        # This helps prevent some of the CPU dispatcher issues
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1" 
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        
        # Debug: Write environment variables
        with open(debug_log_path, 'a') as f:
            f.write("Environment variables set:\n")
            for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", 
                      "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
                f.write(f"  {var}={os.environ.get(var, 'not set')}\n")
        
        # Create a dummy numpy module if running in PyInstaller bundle
        # This helps prevent double-initialization in some cases
        if getattr(sys, '_MEIPASS', None):
            # Only apply in PyInstaller bundle
            with open(debug_log_path, 'a') as f:
                f.write("Running in PyInstaller bundle, applying NumPy fix\n")
            
            # Check for common Python packages in sys.modules
            with open(debug_log_path, 'a') as f:
                f.write("Modules already imported:\n")
                for mod in ["numpy", "llama_cpp", "PyQt5"]:
                    f.write(f"  {mod} in sys.modules: {mod in sys.modules}\n")
                    
            try:
                # Patch the NumPy tracer first
                # Create a monkey patch for numpy.core._multiarray_umath
                # which might help avoid the CPU dispatcher initialization issue
                if 'numpy' not in sys.modules and 'numpy.core._multiarray_umath' not in sys.modules:
                    with open(debug_log_path, 'a') as f:
                        f.write("Creating dummy numpy.core._multiarray_umath to prevent duplicate init\n")
                    
                    # Create a dummy module to prevent double initialization
                    import types
                    dummy_module = types.ModuleType('numpy.core._multiarray_umath')
                    sys.modules['numpy.core._multiarray_umath'] = dummy_module
                    
                    with open(debug_log_path, 'a') as f:
                        f.write("Dummy module created\n")
                
                # Check if numpy is already in sys.modules
                if 'numpy' not in sys.modules:
                    with open(debug_log_path, 'a') as f:
                        f.write("NumPy not yet imported, doing controlled import\n")
                    
                    # Try to safely preload numpy to control initialization
                    import numpy as np
                    
                    with open(debug_log_path, 'a') as f:
                        f.write(f"NumPy imported, version: {np.__version__}\n")
                        f.write("Initializing NumPy config...\n")
                    
                    # Explicitly initialize BLAS and FFT backends to ensure they're only done once
                    try:
                        np.__config__.show()  # This forces initialization of config values
                        with open(debug_log_path, 'a') as f:
                            f.write("NumPy config initialized successfully\n")
                    except Exception as config_error:
                        with open(debug_log_path, 'a') as f:
                            f.write(f"Error initializing NumPy config: {config_error}\n")
                    
                    # Release memory not needed
                    del np
                else:
                    # If numpy is already imported, ensure it's fully initialized
                    with open(debug_log_path, 'a') as f:
                        f.write("NumPy already imported, ensuring full initialization\n")
                    
                    import numpy as np
                    try:
                        np.__config__.show()
                        with open(debug_log_path, 'a') as f:
                            f.write("NumPy config re-initialized successfully\n")
                    except Exception as config_error:
                        with open(debug_log_path, 'a') as f:
                            f.write(f"Error re-initializing NumPy config: {config_error}\n")
                    del np
                
                with open(debug_log_path, 'a') as f:
                    f.write("NumPy fixes applied successfully\n")
                return True
            except Exception as e:
                # Log error but don't crash if this fix fails
                with open(debug_log_path, 'a') as f:
                    f.write(f"ERROR applying NumPy fixes: {str(e)}\n")
                    f.write(traceback.format_exc() + "\n")
                logger.error(f"Error applying NumPy fixes: {str(e)}")
                return False
        else:
            # Not running in PyInstaller, just set environment variables
            with open(debug_log_path, 'a') as f:
                f.write("Not running in PyInstaller bundle, only setting environment variables\n")
            return True
    except Exception as outer_error:
        # Last resort error handling
        try:
            with open(debug_log_path, 'a') as f:
                f.write(f"CRITICAL ERROR in apply_numpy_fixes: {str(outer_error)}\n")
                f.write(traceback.format_exc() + "\n")
        except:
            pass
        return False
