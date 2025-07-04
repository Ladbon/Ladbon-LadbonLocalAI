import os
import sys
import shutil
import logging
from datetime import datetime
import re

# Set up logging
log_file = f"cuda_patch_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("=" * 50)
logging.info("CUDA Patch Script for Packaged App")
logging.info("=" * 50)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGED_APP_DIR = os.path.join(BASE_DIR, "dist", "Ladbon AI Desktop")
INTERNAL_DIR = os.path.join(PACKAGED_APP_DIR, "_internal")
UTILS_DIR = os.path.join(INTERNAL_DIR, "utils")
LLAMACPP_LOADER_PATH = os.path.join(UTILS_DIR, "llamacpp_loader.py")
INIT_CUDA_PATH = os.path.join(UTILS_DIR, "init_cuda.py")

# Check if directories exist
if not os.path.exists(PACKAGED_APP_DIR):
    logging.error(f"Packaged app directory not found: {PACKAGED_APP_DIR}")
    sys.exit(1)

if not os.path.exists(UTILS_DIR):
    logging.warning(f"Utils directory not found: {UTILS_DIR}")
    logging.info("Creating utils directory...")
    os.makedirs(UTILS_DIR, exist_ok=True)

# Create or update init_cuda.py
logging.info(f"Creating {INIT_CUDA_PATH}...")
with open(INIT_CUDA_PATH, "w") as f:
    f.write("""# CUDA initialization module
import os
import sys
import logging
import ctypes
from pathlib import Path

def init_cuda_environment():
    """Initialize CUDA environment by ensuring DLLs are in PATH"""
    logger = logging.getLogger("cuda_init")
    
    # Get the application directory
    app_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    base_dir = Path(app_dir)
    
    # Look for cuda_dlls directory
    cuda_dirs = [
        base_dir / "cuda_dlls",
        base_dir.parent / "cuda_dlls",
        base_dir / "_internal" / "cuda_dlls",
        Path(os.path.abspath(".")) / "cuda_dlls",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin"),
        Path("C:/Windows/System32")
    ]
    
    # Log current PATH
    logger.info(f"Current PATH: {os.environ.get('PATH', '')}")
    
    # Check each directory for CUDA DLLs
    cuda_dlls_found = False
    cuda_paths_added = []
    
    for cuda_dir in cuda_dirs:
        if cuda_dir.exists():
            logger.info(f"Checking directory: {cuda_dir}")
            dlls = list(cuda_dir.glob("*cuda*.dll")) + list(cuda_dir.glob("*cublas*.dll"))
            
            if dlls:
                cuda_dlls_found = True
                logger.info(f"Found {len(dlls)} CUDA DLLs in {cuda_dir}")
                for dll in dlls:
                    logger.info(f"  - {dll.name}")
                
                # Add to PATH if not already there
                cuda_dir_str = str(cuda_dir)
                if cuda_dir_str not in os.environ.get('PATH', ''):
                    logger.info(f"Adding to PATH: {cuda_dir_str}")
                    os.environ['PATH'] = cuda_dir_str + os.pathsep + os.environ.get('PATH', '')
                    cuda_paths_added.append(cuda_dir_str)
    
    if not cuda_dlls_found:
        logger.warning("No CUDA DLLs found in any of the search directories!")
    else:
        logger.info(f"Added {len(cuda_paths_added)} directories to PATH")
    
    # Directly try to load important CUDA DLLs
    cuda_dlls = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
    for dll in cuda_dlls:
        try:
            lib = ctypes.CDLL(dll)
            logger.info(f"Successfully loaded {dll}")
        except Exception as e:
            logger.warning(f"Failed to load {dll}: {e}")
    
    return cuda_dlls_found
""")
logging.info(f"Created {INIT_CUDA_PATH}")

# Check if llamacpp_loader.py exists
if not os.path.exists(LLAMACPP_LOADER_PATH):
    logging.warning(f"llamacpp_loader.py not found at {LLAMACPP_LOADER_PATH}")
    logging.info("Creating basic llamacpp_loader.py...")
    
    # Create a basic llamacpp_loader.py if it doesn't exist
    with open(LLAMACPP_LOADER_PATH, "w") as f:
        f.write("""# Enhanced LlamaCpp loader with CUDA support
import os
import sys
import logging
from pathlib import Path
import importlib.util
import ctypes

# Configure logging
logger = logging.getLogger("llamacpp_loader")
logger.setLevel(logging.DEBUG)

# Try to import init_cuda
try:
    from utils.init_cuda import init_cuda_environment
    init_cuda_environment()
    logger.info("CUDA environment initialized")
except Exception as e:
    logger.warning(f"Failed to initialize CUDA environment: {e}")

def load_llamacpp(use_cuda=True):
    """Load llama_cpp with CUDA support if available"""
    logger.info(f"Loading llama_cpp with CUDA={use_cuda}")
    
    try:
        # First try to import normally
        import llama_cpp
        logger.info("Successfully imported llama_cpp")
        
        # Try to initialize the CUDA backend if requested
        if use_cuda:
            try:
                logger.info("Initializing CUDA backend...")
                # Try different known initialization methods
                try:
                    llama_cpp.llama_backend_init(True)
                    logger.info("CUDA backend initialized with llama_backend_init(True)")
                except (AttributeError, TypeError) as e1:
                    logger.warning(f"Method 1 failed: {e1}")
                    try:
                        llama_cpp.llama_backend_init(use_cuda=True)
                        logger.info("CUDA backend initialized with llama_backend_init(use_cuda=True)")
                    except (AttributeError, TypeError) as e2:
                        logger.warning(f"Method 2 failed: {e2}")
                        if hasattr(llama_cpp, '_lib') and hasattr(llama_cpp._lib, 'llama_backend_init'):
                            llama_cpp._lib.llama_backend_init()
                            logger.info("CUDA backend initialized with _lib.llama_backend_init()")
                        else:
                            logger.error("Could not find llama_backend_init method")
            except Exception as e:
                logger.error(f"Failed to initialize CUDA backend: {e}")
                logger.info("Falling back to CPU-only mode")
        
        return llama_cpp
    
    except Exception as e:
        logger.error(f"Failed to import llama_cpp: {e}")
        return None
""")
    logging.info("Created basic llamacpp_loader.py")
else:
    # Patch existing llamacpp_loader.py
    logging.info(f"Patching {LLAMACPP_LOADER_PATH}...")
    
    with open(LLAMACPP_LOADER_PATH, "r") as f:
        content = f.read()
    
    # Check if init_cuda import is already present
    if "from utils.init_cuda import init_cuda_environment" not in content:
        # Add import and initialization
        content = re.sub(
            r"(import\s+.*?)(\n\n|\n# )",
            r"\1\n\n# Import CUDA environment initializer\ntry:\n    from utils.init_cuda import init_cuda_environment\n    init_cuda_environment()\n    logger.info(\"CUDA environment initialized\")\nexcept Exception as e:\n    logger.warning(f\"Failed to initialize CUDA environment: {e}\")\n\n",
            content,
            count=1
        )
    
    # Update or add load_llamacpp function with robust CUDA initialization
    if "def load_llamacpp" in content:
        # Replace existing function with enhanced version
        content = re.sub(
            r"def load_llamacpp.*?(?=\ndef|\Z)",
            """def load_llamacpp(use_cuda=True):
    \"\"\"Load llama_cpp with CUDA support if available\"\"\"
    logger.info(f"Loading llama_cpp with CUDA={use_cuda}")
    
    try:
        # First try to import normally
        import llama_cpp
        logger.info("Successfully imported llama_cpp")
        
        # Try to initialize the CUDA backend if requested
        if use_cuda:
            try:
                logger.info("Initializing CUDA backend...")
                # Try different known initialization methods
                try:
                    llama_cpp.llama_backend_init(True)
                    logger.info("CUDA backend initialized with llama_backend_init(True)")
                except (AttributeError, TypeError) as e1:
                    logger.warning(f"Method 1 failed: {e1}")
                    try:
                        llama_cpp.llama_backend_init(use_cuda=True)
                        logger.info("CUDA backend initialized with llama_backend_init(use_cuda=True)")
                    except (AttributeError, TypeError) as e2:
                        logger.warning(f"Method 2 failed: {e2}")
                        if hasattr(llama_cpp, '_lib') and hasattr(llama_cpp._lib, 'llama_backend_init'):
                            llama_cpp._lib.llama_backend_init()
                            logger.info("CUDA backend initialized with _lib.llama_backend_init()")
                        else:
                            logger.error("Could not find llama_backend_init method")
            except Exception as e:
                logger.error(f"Failed to initialize CUDA backend: {e}")
                logger.info("Falling back to CPU-only mode")
        
        return llama_cpp
    
    except Exception as e:
        logger.error(f"Failed to import llama_cpp: {e}")
        return None
""",
            content,
            flags=re.DOTALL
        )
    else:
        # Add the function if it doesn't exist
        content += "\n\n" + """def load_llamacpp(use_cuda=True):
    \"\"\"Load llama_cpp with CUDA support if available\"\"\"
    logger.info(f"Loading llama_cpp with CUDA={use_cuda}")
    
    try:
        # First try to import normally
        import llama_cpp
        logger.info("Successfully imported llama_cpp")
        
        # Try to initialize the CUDA backend if requested
        if use_cuda:
            try:
                logger.info("Initializing CUDA backend...")
                # Try different known initialization methods
                try:
                    llama_cpp.llama_backend_init(True)
                    logger.info("CUDA backend initialized with llama_backend_init(True)")
                except (AttributeError, TypeError) as e1:
                    logger.warning(f"Method 1 failed: {e1}")
                    try:
                        llama_cpp.llama_backend_init(use_cuda=True)
                        logger.info("CUDA backend initialized with llama_backend_init(use_cuda=True)")
                    except (AttributeError, TypeError) as e2:
                        logger.warning(f"Method 2 failed: {e2}")
                        if hasattr(llama_cpp, '_lib') and hasattr(llama_cpp._lib, 'llama_backend_init'):
                            llama_cpp._lib.llama_backend_init()
                            logger.info("CUDA backend initialized with _lib.llama_backend_init()")
                        else:
                            logger.error("Could not find llama_backend_init method")
            except Exception as e:
                logger.error(f"Failed to initialize CUDA backend: {e}")
                logger.info("Falling back to CPU-only mode")
        
        return llama_cpp
    
    except Exception as e:
        logger.error(f"Failed to import llama_cpp: {e}")
        return None
"""
    
    # Write the updated content back
    with open(LLAMACPP_LOADER_PATH, "w") as f:
        f.write(content)
    
    logging.info(f"Successfully patched {LLAMACPP_LOADER_PATH}")

# Create a "Create CUDA Import Hook" script
logging.info("Creating CUDA import hook...")
CUDA_HOOK_PATH = os.path.join(INTERNAL_DIR, "cuda_import_hook.py")
with open(CUDA_HOOK_PATH, "w") as f:
    f.write("""# CUDA Import Hook
import os
import sys
import logging
from pathlib import Path
import ctypes

# Set up logging
logger = logging.getLogger("cuda_hook")
log_file = "logs/cuda_hook.log"
os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

logger.info("=" * 50)
logger.info("CUDA Import Hook Activated")
logger.info("=" * 50)

def initialize_cuda_environment():
    """Make sure CUDA DLLs can be found and loaded"""
    # Get the application directory
    app_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    base_dir = Path(app_dir)
    
    # Look for cuda_dlls directory
    cuda_dirs = [
        base_dir / "cuda_dlls",
        base_dir.parent / "cuda_dlls",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin")
    ]
    
    # Log current PATH
    logger.info(f"Current PATH: {os.environ.get('PATH', '')}")
    
    # Add CUDA directories to PATH
    for cuda_dir in cuda_dirs:
        if cuda_dir.exists():
            logger.info(f"Found CUDA directory: {cuda_dir}")
            
            # Look for CUDA DLLs
            dlls = list(cuda_dir.glob("*cuda*.dll")) + list(cuda_dir.glob("*cublas*.dll"))
            if dlls:
                logger.info(f"Found {len(dlls)} CUDA DLLs in {cuda_dir}")
                for dll in dlls[:5]:  # List first 5 DLLs
                    logger.info(f"  - {dll.name}")
                
                # Add to PATH if not already there
                cuda_dir_str = str(cuda_dir)
                if cuda_dir_str not in os.environ.get('PATH', ''):
                    logger.info(f"Adding to PATH: {cuda_dir_str}")
                    os.environ['PATH'] = cuda_dir_str + os.pathsep + os.environ.get('PATH', '')

# Class to hook llama_cpp import
class LlamaCppImportHook:
    def __init__(self):
        self.original_import = __import__
    
    def custom_import(self, name, *args, **kwargs):
        module = self.original_import(name, *args, **kwargs)
        
        # Hook llama_cpp import
        if name == 'llama_cpp':
            logger.info(f"Intercepted import of {name}")
            try:
                # Try to pre-load CUDA DLLs
                cuda_dlls = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
                for dll in cuda_dlls:
                    try:
                        lib = ctypes.CDLL(dll)
                        logger.info(f"Successfully pre-loaded {dll}")
                    except Exception as e:
                        logger.warning(f"Failed to pre-load {dll}: {e}")
                
                # Patch module's backend init if needed
                if hasattr(module, 'llama_backend_init'):
                    original_init = module.llama_backend_init
                    
                    def patched_init(use_cuda=False, **kwargs):
                        logger.info(f"Intercepted llama_backend_init(use_cuda={use_cuda}, kwargs={kwargs})")
                        try:
                            result = original_init(use_cuda, **kwargs)
                            logger.info("llama_backend_init succeeded")
                            return result
                        except Exception as e:
                            logger.error(f"llama_backend_init failed: {e}")
                            logger.info("Retrying with different parameters...")
                            
                            try:
                                if hasattr(module, '_lib') and hasattr(module._lib, 'llama_backend_init'):
                                    module._lib.llama_backend_init()
                                    logger.info("_lib.llama_backend_init succeeded")
                                    return True
                            except Exception as e2:
                                logger.error(f"_lib.llama_backend_init failed: {e2}")
                                return False
                    
                    module.llama_backend_init = patched_init
                    logger.info("Successfully patched llama_backend_init")
            
            except Exception as e:
                logger.error(f"Error during llama_cpp import patching: {e}")
        
        return module

# Install the import hook
initialize_cuda_environment()
sys.meta_path.insert(0, LlamaCppImportHook())
logger.info("CUDA import hook installed")
""")
logging.info(f"Created {CUDA_HOOK_PATH}")

# Create a startup script to inject the hook
logging.info("Creating startup script...")
STARTUP_HOOK_PATH = os.path.join(INTERNAL_DIR, "sitecustomize.py")
with open(STARTUP_HOOK_PATH, "w") as f:
    f.write("""# Site customization for CUDA support
import os
import sys
import logging
from pathlib import Path

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/sitecustomize.log",
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("sitecustomize")

logger.info("=" * 50)
logger.info("Site customization for CUDA support")
logger.info("=" * 50)

# Add CUDA directories to PATH
try:
    # Get the application directory
    app_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    base_dir = Path(app_dir)
    
    # Look for cuda_dlls directory and add to PATH
    cuda_dirs = [
        base_dir / "cuda_dlls",
        base_dir.parent / "cuda_dlls",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin")
    ]
    
    for cuda_dir in cuda_dirs:
        if cuda_dir.exists():
            logger.info(f"Found CUDA directory: {cuda_dir}")
            cuda_dir_str = str(cuda_dir)
            if cuda_dir_str not in os.environ.get('PATH', ''):
                logger.info(f"Adding to PATH: {cuda_dir_str}")
                os.environ['PATH'] = cuda_dir_str + os.pathsep + os.environ.get('PATH', '')
    
    # Import the CUDA hook
    try:
        import cuda_import_hook
        logger.info("CUDA import hook loaded")
    except ImportError as e:
        logger.warning(f"Could not import cuda_import_hook: {e}")
except Exception as e:
    logger.error(f"Error in sitecustomize.py: {e}")

logger.info("Site customization completed")
""")
logging.info(f"Created {STARTUP_HOOK_PATH}")

# Create new Python launcher
logging.info("Creating Python launcher...")
PYTHON_LAUNCHER_PATH = os.path.join(PACKAGED_APP_DIR, "Launch_with_Python_CUDA.bat")
with open(PYTHON_LAUNCHER_PATH, "w") as f:
    f.write("""@echo off
setlocal EnableDelayedExpansion

:: Set up logging
set LOG_FILE=logs\python_launcher_%date:~-4,4%%date:~-7,2%%date:~-10,2%-%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

if not exist logs mkdir logs

echo ================================================ > "%LOG_FILE%"
echo Python-CUDA launcher for Ladbon AI Desktop >> "%LOG_FILE%"
echo Started at: %date% %time% >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"

:: Add CUDA DLLs directory to PATH
set ORIGINAL_PATH=%PATH%
echo Original PATH: %PATH% >> "%LOG_FILE%"

echo Adding CUDA paths to PATH... >> "%LOG_FILE%"
set PATH=%~dp0_internal\cuda_dlls;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
echo New PATH: %PATH% >> "%LOG_FILE%"

:: Create and run a Python script to load the GUI directly
echo Creating Python launcher script... >> "%LOG_FILE%"
echo import sys, os >> "%~dp0_internal\python_launcher.py"
echo import logging >> "%~dp0_internal\python_launcher.py"
echo logging.basicConfig(level=logging.DEBUG, filename='logs/python_app.log', filemode='w', format='%%(asctime)s [%%(levelname)s] %%(message)s') >> "%~dp0_internal\python_launcher.py"
echo logging.info('Python launcher starting') >> "%~dp0_internal\python_launcher.py"
echo sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '_internal')) >> "%~dp0_internal\python_launcher.py"
echo logging.info('Python paths: ' + str(sys.path)) >> "%~dp0_internal\python_launcher.py"
echo try: >> "%~dp0_internal\python_launcher.py"
echo     import cuda_import_hook >> "%~dp0_internal\python_launcher.py"
echo     logging.info('CUDA hook imported') >> "%~dp0_internal\python_launcher.py"
echo except ImportError as e: >> "%~dp0_internal\python_launcher.py"
echo     logging.warning(f'Could not import CUDA hook: {e}') >> "%~dp0_internal\python_launcher.py"
echo logging.info('Importing main module') >> "%~dp0_internal\python_launcher.py"
echo import api.app >> "%~dp0_internal\python_launcher.py"
echo logging.info('Starting app') >> "%~dp0_internal\python_launcher.py"
echo api.app.main() >> "%~dp0_internal\python_launcher.py"

:: Run the launcher script
echo Running Python launcher script... >> "%LOG_FILE%"
"%~dp0_internal\python.exe" "%~dp0_internal\python_launcher.py" --use-cuda --verbose

echo Launcher exited. >> "%LOG_FILE%"
endlocal
""")
logging.info(f"Created {PYTHON_LAUNCHER_PATH}")

# Create new Direct CUDA launcher
logging.info("Creating direct CUDA launcher...")
DIRECT_LAUNCHER_PATH = os.path.join(PACKAGED_APP_DIR, "Launch_with_Direct_CUDA_Support.bat")
with open(DIRECT_LAUNCHER_PATH, "w") as f:
    f.write("""@echo off
setlocal EnableDelayedExpansion

:: Set up logging
set LOG_FILE=logs\direct_cuda_launcher_%date:~-4,4%%date:~-7,2%%date:~-10,2%-%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

if not exist logs mkdir logs

echo ================================================ > "%LOG_FILE%"
echo Direct CUDA launcher for Ladbon AI Desktop >> "%LOG_FILE%"
echo Started at: %date% %time% >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"

:: Set up CUDA environment variables
set ORIGINAL_PATH=%PATH%
echo Original PATH: %PATH% >> "%LOG_FILE%"

echo Setting CUDA environment variables... >> "%LOG_FILE%"
set PATH=%~dp0_internal\cuda_dlls;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%
set PYTHONPATH=%~dp0_internal
echo Modified PATH: %PATH% >> "%LOG_FILE%"
echo PYTHONPATH: %PYTHONPATH% >> "%LOG_FILE%"

:: Check for CUDA DLLs in PATH
echo Checking for CUDA DLLs in PATH... >> "%LOG_FILE%"
where cudart64_12.dll >> "%LOG_FILE%" 2>&1
where cublas64_12.dll >> "%LOG_FILE%" 2>&1

:: Create a CUDA test script
echo Creating CUDA test script... >> "%LOG_FILE%"
if not exist "%~dp0_internal\test_cuda.py" (
    echo import os, sys, ctypes >> "%~dp0_internal\test_cuda.py"
    echo print("Python executable:", sys.executable) >> "%~dp0_internal\test_cuda.py"
    echo print("PYTHONPATH:", os.environ.get('PYTHONPATH', '')) >> "%~dp0_internal\test_cuda.py"
    echo print("PATH:", os.environ.get('PATH', '')) >> "%~dp0_internal\test_cuda.py"
    echo print("\nTesting CUDA DLLs:") >> "%~dp0_internal\test_cuda.py"
    echo for dll in ["cudart64_12.dll", "cublas64_12.dll"]: >> "%~dp0_internal\test_cuda.py"
    echo     try: >> "%~dp0_internal\test_cuda.py"
    echo         lib = ctypes.CDLL(dll) >> "%~dp0_internal\test_cuda.py"
    echo         print(f"  ✓ {dll} loaded successfully") >> "%~dp0_internal\test_cuda.py"
    echo     except Exception as e: >> "%~dp0_internal\test_cuda.py"
    echo         print(f"  ✗ {dll} failed to load: {e}") >> "%~dp0_internal\test_cuda.py"
    echo print("\nTesting llama_cpp:") >> "%~dp0_internal\test_cuda.py"
    echo try: >> "%~dp0_internal\test_cuda.py"
    echo     import llama_cpp >> "%~dp0_internal\test_cuda.py"
    echo     print(f"  ✓ llama_cpp imported successfully") >> "%~dp0_internal\test_cuda.py"
    echo     print(f"  Version: {getattr(llama_cpp, '__version__', 'unknown')}") >> "%~dp0_internal\test_cuda.py"
    echo     print("  Testing llama_backend_init...") >> "%~dp0_internal\test_cuda.py"
    echo     try: >> "%~dp0_internal\test_cuda.py"
    echo         llama_cpp.llama_backend_init(True) >> "%~dp0_internal\test_cuda.py"
    echo         print("  ✓ CUDA backend initialized successfully") >> "%~dp0_internal\test_cuda.py"
    echo     except Exception as e: >> "%~dp0_internal\test_cuda.py"
    echo         print(f"  ✗ CUDA backend initialization failed: {e}") >> "%~dp0_internal\test_cuda.py"
    echo except Exception as e: >> "%~dp0_internal\test_cuda.py"
    echo     print(f"  ✗ llama_cpp import failed: {e}") >> "%~dp0_internal\test_cuda.py"
)

:: Run the test script
echo Running CUDA test script... >> "%LOG_FILE%"
"%~dp0_internal\python.exe" "%~dp0_internal\test_cuda.py" >> "%LOG_FILE%" 2>&1

:: Launch the app with CUDA support
echo Launching the app with CUDA support... >> "%LOG_FILE%"
"%~dp0Ladbon AI Desktop.exe" --use-cuda --verbose

echo Launcher exited. >> "%LOG_FILE%"
endlocal
""")
logging.info(f"Created {DIRECT_LAUNCHER_PATH}")

logging.info("=" * 50)
logging.info("CUDA patch completed")
logging.info("=" * 50)
print("\nCUDA patch completed successfully! Check logs for details.")
