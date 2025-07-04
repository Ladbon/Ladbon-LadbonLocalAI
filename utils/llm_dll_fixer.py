"""
Utility module for fixing llama-cpp-python DLL loading issues in packaged applications
This implements safer patch and loading mechanisms for handling the DLLs
"""
import os
import sys
import ctypes
import platform
import logging
from pathlib import Path

logger = logging.getLogger('llm_dll_fixer')

def setup_dll_paths():
    """
    Set up DLL search paths for Windows to find all required DLLs.
    This needs to be called before importing llama_cpp.
    """
    if platform.system() != "Windows":
        logger.info("DLL path setup only needed on Windows")
        return True
    
    # For packaged apps, add DLL directories to PATH
    if getattr(sys, 'frozen', False):
        executable_dir = os.path.dirname(sys.executable)
        # Add possible DLL locations to path
        dll_paths = [
            os.path.join(executable_dir, '_internal', 'llama_cpp', 'lib'),
            os.path.join(executable_dir, '_internal', 'llama_cpp'),
            os.path.join(executable_dir, '_internal')
        ]
        
        # Log current PATH
        logger.info(f"Current PATH: {os.environ.get('PATH', '')}")
        
        # Add our paths to the front of PATH
        for path in dll_paths:
            if os.path.exists(path):
                logger.info(f"Adding to PATH: {path}")
                os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
        
        # Also try to use AddDllDirectory for Windows 8+ systems
        try:
            for path in dll_paths:
                if os.path.exists(path):
                    # Convert to unicode for AddDllDirectory
                    if isinstance(path, bytes):
                        path = path.decode('utf-8')
                    logger.info(f"Adding DLL directory: {path}")
                    ctypes.windll.kernel32.AddDllDirectory(path)
        except Exception as dll_ex:
            logger.warning(f"AddDllDirectory failed: {dll_ex}")
    
    return True

def find_dll_path(dll_name):
    """
    Find a specific DLL in the expected paths
    """
    if platform.system() != "Windows":
        return None
    
    # Get the directory of the executable
    if getattr(sys, 'frozen', False):
        # We're running as a bundled exe
        base_path = sys.executable
        exe_dir = os.path.dirname(base_path)
    else:
        # We're running in a normal Python environment
        base_path = sys.executable
        exe_dir = os.path.dirname(base_path)
    
    # Potential DLL locations
    dll_locations = [
        exe_dir,
        os.path.join(exe_dir, '_internal'),
        os.path.join(exe_dir, '_internal', 'llama_cpp'),
        os.path.join(exe_dir, '_internal', 'llama_cpp', 'lib'),
    ]
    
    # Try to find the DLL
    for location in dll_locations:
        dll_path = os.path.join(location, dll_name)
        if os.path.exists(dll_path):
            logger.info(f"Found {dll_name} at: {dll_path}")
            return dll_path
    
    logger.warning(f"Could not find {dll_name} in any of the expected locations")
    return None

def patch_backend_initialization():
    """
    Patch the llama_cpp module to use a more robust backend initialization
    """
    try:
        import llama_cpp
        logger.info(f"Successfully imported llama_cpp from {llama_cpp.__file__}")
        
        # Save original functions to restore if needed
        original_backend_init = getattr(llama_cpp, 'llama_backend_init', None)
        
        # Find the actual DLL path
        llama_dll_path = None
        if platform.system() == "Windows":
            # Try to get the DLL path from multiple locations
            possible_locations = [
                os.path.join(os.path.dirname(llama_cpp.__file__), 'lib', 'llama.dll'),
                os.path.join(os.path.dirname(llama_cpp.__file__), 'llama.dll'),
            ]
            
            if getattr(sys, 'frozen', False):
                # If running as frozen app, add PyInstaller-specific paths
                exe_dir = os.path.dirname(sys.executable)
                possible_locations.extend([
                    os.path.join(exe_dir, '_internal', 'llama_cpp', 'lib', 'llama.dll'),
                    os.path.join(exe_dir, '_internal', 'llama_cpp', 'llama.dll'),
                    os.path.join(exe_dir, '_internal', 'llama.dll'),
                ])
            
            # Find the first DLL that exists
            for path in possible_locations:
                if os.path.exists(path):
                    llama_dll_path = path
                    logger.info(f"Found llama.dll at: {llama_dll_path}")
                    break
        
        # If DLL not found with direct paths, try searching
        if not llama_dll_path:
            logger.info("Searching for llama.dll...")
            llama_dll_path = find_dll_path("llama.dll")
        
        if not llama_dll_path:
            logger.error("Could not find llama.dll in any expected location")
            return False
        
        # Load the DLL directly
        try:
            logger.info(f"Loading DLL directly from: {llama_dll_path}")
            llama_dll = ctypes.CDLL(llama_dll_path)
            
            # Check available functions in the DLL
            dll_functions = []
            if hasattr(llama_dll, 'llama_backend_init'):
                dll_functions.append('llama_backend_init')
                # Define the function signature
                llama_dll.llama_backend_init.argtypes = [ctypes.c_bool]
                llama_dll.llama_backend_init.restype = None
            
            if hasattr(llama_dll, 'llama_init_backend'):
                dll_functions.append('llama_init_backend')
                # Define the function signature
                llama_dll.llama_init_backend.argtypes = [ctypes.c_bool]
                llama_dll.llama_init_backend.restype = None
            
            logger.info(f"Found functions in DLL: {dll_functions}")
            
            # Define a robust backend init function that tries both function names
            def robust_backend_init():
                """Robust backend initialization that tries multiple function names"""
                logger.debug("Calling robust backend init")
                if hasattr(llama_dll, 'llama_backend_init'):
                    logger.debug("Calling llama_backend_init")
                    # Call without arguments to be safe
                    llama_dll.llama_backend_init()
                    return True
                elif hasattr(llama_dll, 'llama_init_backend'):
                    logger.debug("Calling llama_init_backend")
                    # Call without arguments to be safe
                    llama_dll.llama_init_backend()
                    return True
                else:
                    logger.error("No backend initialization function found in DLL")
                    return False
            
            # Apply the patch to llama_cpp module
            llama_cpp.llama_backend_init = robust_backend_init
            
            # Test if it works
            try:
                logger.info("Testing robust backend initialization")
                # Check if the function expects parameters by inspecting its signature
                import inspect
                # Always call it without parameters to be safe
                logger.info("Calling llama_backend_init without parameters")
                llama_cpp.llama_backend_init()
                logger.info("Successfully initialized llama_cpp backend")
                return True
            except Exception as e:
                logger.error(f"Error testing robust backend initialization: {e}")
                # Restore original if needed
                if original_backend_init is not None:
                    llama_cpp.llama_backend_init = original_backend_init
                return False
        
        except Exception as e:
            logger.error(f"Failed to load or initialize DLL: {e}")
            return False
    
    except ImportError as e:
        logger.error(f"Cannot import llama_cpp: {e}")
        return False
    except Exception as e:
        logger.error(f"Error patching llama_cpp backend: {e}", exc_info=True)
        return False

def safe_backend_init():
    """
    Safely initialize the llama_cpp backend with appropriate error handling
    """
    try:
        # First try to patch the initialization function
        patch_success = patch_backend_initialization()
        if patch_success:
            logger.info("Successfully patched and initialized llama_cpp backend")
            return True
        
        # If patching failed, try the old approaches
        import llama_cpp
        
        # Try different approaches to initialize the backend
        
        # Approach 1: Use the direct llama_backend_init function if available
        if hasattr(llama_cpp, 'llama_backend_init'):
            try:
                logger.info("Initializing llama_cpp backend using llama_backend_init")
                # Always call without parameters to be safe
                logger.info("Calling llama_backend_init without parameters")
                llama_cpp.llama_backend_init()
                logger.info("Successfully initialized llama_cpp backend")
                return True
            except Exception as e:
                logger.warning(f"Failed to initialize backend with llama_backend_init: {e}")
        
        # Log warning about failure
        logger.warning("All backend initialization approaches failed")
        return False
    
    except ImportError as e:
        logger.error(f"Cannot import llama_cpp: {e}")
        return False
    except Exception as e:
        logger.error(f"Error initializing llama_cpp backend: {e}", exc_info=True)
        return False
        
def create_runtime_patcher():
    """
    Create a runtime patcher for the llama-cpp-python module to make it work in packaged applications
    """
    try:
        import llama_cpp
        logger.info("Setting up runtime patching for llama-cpp-python")
        
        # Check if Llama class exists and grab its original __init__
        if not hasattr(llama_cpp, 'Llama'):
            logger.warning("Llama class not found in llama_cpp module")
            return False
        
        # Store the original __init__ method
        original_llama_init = llama_cpp.Llama.__init__
        
        def patched_llama_init(self, *args, **kwargs):
            """
            Patched __init__ method for the Llama class that handles backend initialization
            """
            logger.info("Using patched Llama.__init__")
            
            # Try to initialize the backend first
            try:
                # Use a more compatible way to track initialization status
                backend_initialized = getattr(llama_cpp, '_backend_initialized', False)
                if not backend_initialized:
                    logger.info("Backend not yet initialized, calling patch_backend_initialization")
                    success = patch_backend_initialization()
                    if success:
                        # Set a flag on the module in a safer way
                        setattr(llama_cpp, '_backend_initialized', True)
                    else:
                        # Try safe_backend_init as fallback
                        logger.info("Trying safe_backend_init as fallback")
                        success = safe_backend_init()
                        if success:
                            setattr(llama_cpp, '_backend_initialized', True)
                
                # Now call the original __init__
                logger.info("Calling original Llama.__init__")
                original_llama_init(self, *args, **kwargs)
                logger.info("Original Llama.__init__ completed successfully")
            
            except Exception as e:
                logger.error(f"Error in patched Llama.__init__: {e}")
                raise
        
        # Apply the patch (we already checked if Llama class exists earlier)
        logger.info("Applying patch to Llama class")
        llama_cpp.Llama.__init__ = patched_llama_init
        return True
    
    except ImportError:
        logger.error("Cannot import llama_cpp module")
        return False
    except Exception as e:
        logger.error(f"Error setting up runtime patching: {e}")
        return False

def initialize():
    """
    Complete initialization process for llama-cpp in packaged applications
    """
    # 1. Set up DLL paths first
    path_setup = setup_dll_paths()
    if not path_setup:
        logger.error("Failed to set up DLL paths")
        return False
    
    # 2. Initialize the backend if directly importing llama_cpp now
    backend_init = safe_backend_init()
    if not backend_init:
        logger.warning("Direct backend initialization failed - will try runtime patching")
    
    # 3. Set up runtime patching for later model loading
    patcher_result = create_runtime_patcher()
    if patcher_result:
        logger.info("Runtime patcher for llama-cpp-python set up successfully")
    else:
        logger.warning("Failed to set up runtime patcher - may have issues loading models")
    
    # Return True even if some steps failed - we have multiple approaches
    return True

if __name__ == "__main__":
    # Configure logging when run directly
    logging.basicConfig(level=logging.INFO)
    initialize()
