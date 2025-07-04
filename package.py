import os
import shutil
import sys
import subprocess
import PyInstaller.__main__
import platform
import glob
import ctypes.util
from pathlib import Path

# Check and install required packages if needed
try:
    import PyInstaller
except ImportError:
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

try:
    import psutil
except ImportError:
    print("Installing psutil (required for llamacpp_client)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])

# Define paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SRC_DIR
APP_NAME = "Ladbon AI Desktop"

# Move to the correct directory
os.chdir(SRC_DIR)

# CUDA dependency check functions
def find_cuda_dlls():
    """Find all CUDA DLLs in the system PATH"""
    cuda_dlls = []
    paths = os.environ.get("PATH", "").split(os.pathsep)
    
    # Add known CUDA paths
    known_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files\NVIDIA Corporation"
    ]
    
    # Find all CUDA version directories
    for base_path in known_paths:
        if os.path.exists(base_path):
            for cuda_ver in glob.glob(os.path.join(base_path, "*")):
                if os.path.isdir(cuda_ver):
                    bin_path = os.path.join(cuda_ver, "bin")
                    if os.path.exists(bin_path) and os.path.isdir(bin_path):
                        paths.append(bin_path)
    
    print("\n=== CUDA DLL DETECTION ===")
    print(f"Searching for CUDA DLLs in {len(paths)} directories...")
    
    # Find all cudart and cublas DLLs in PATH
    for path in paths:
        if os.path.exists(path):
            for dll_name in ["cudart*.dll", "cublas*.dll", "cudnn*.dll"]:
                for dll_path in glob.glob(os.path.join(path, dll_name)):
                    cuda_dlls.append(dll_path)
    
    # Group DLLs by version
    cuda_versions = {}
    for dll_path in cuda_dlls:
        dll_name = os.path.basename(dll_path)
        # Extract version from name like cudart64_120.dll -> 12.0
        version = None
        if "_" in dll_name:
            parts = dll_name.split("_")
            if len(parts) > 1 and parts[-1].startswith("1"):
                try:
                    ver_num = parts[-1].split(".")[0]
                    major = ver_num[0:2] if len(ver_num) >= 3 else ver_num[0:1]
                    minor = ver_num[2:3] if len(ver_num) >= 3 else "0"
                    version = f"{int(major)}.{int(minor)}"
                except (ValueError, IndexError):
                    pass
        
        if not version:
            version = "unknown"
            
        if version not in cuda_versions:
            cuda_versions[version] = []
        cuda_versions[version].append((dll_name, dll_path))
    
    return cuda_versions

def detect_llama_cpp_cuda_version():
    """Try to detect the CUDA version that llama-cpp-python was built for"""
    try:
        import llama_cpp
        # Try to access the module to see if it's properly installed
        lib_dir = None
        
        # Get the llama_cpp package path
        llamacpp_path = os.path.dirname(llama_cpp.__file__)
        lib_path = os.path.join(llamacpp_path, "lib")
        
        if os.path.exists(lib_path):
            lib_dir = lib_path
        
        if not lib_dir:
            return "unknown"
            
        # Find all DLLs in the directory
        dll_files = glob.glob(os.path.join(lib_dir, "*.dll"))
        
        # Look for CUDA dependencies
        required_dlls = []
        cuda_version = None
        
        # Use a DLL walker or simple heuristic to find dependencies
        for dll_path in dll_files:
            dll_name = os.path.basename(dll_path)
            if "cuda" in dll_name.lower() or "ggml" in dll_name.lower():
                # Try to find what CUDA version this DLL requires
                try:
                    # Load the DLL and use ctypes to get the dependencies
                    # This is a simplified approach - a real DLL walker would be more accurate
                    dll = ctypes.WinDLL(dll_path)
                    # If we got here, the DLL loaded successfully
                    print(f"Successfully loaded: {dll_name}")
                except Exception as e:
                    print(f"Error loading {dll_name}: {str(e)}")
                    
                    # Check if the error message contains a missing DLL name
                    error_str = str(e)
                    if "cudart64_" in error_str or "cublas64_" in error_str:
                        # Extract the DLL name from the error
                        try:
                            import re
                            dll_match = re.search(r'cudart64_(\d+)\.dll|cublas64_(\d+)\.dll', error_str)
                            if dll_match:
                                dll_found = dll_match.group(0)
                                required_dlls.append(dll_found)
                                version_num = dll_match.group(1) or dll_match.group(2)
                                if version_num:
                                    major = version_num[0:2] if len(version_num) >= 3 else version_num[0:1]
                                    minor = version_num[2:3] if len(version_num) >= 3 else "0"
                                    cuda_version = f"{int(major)}.{int(minor)}"
                        except Exception as parse_err:
                            print(f"Error parsing DLL name: {parse_err}")
        
        if cuda_version:
            return cuda_version, required_dlls
        
        # If we couldn't determine from errors, try to infer from filenames
        for dll_path in dll_files:
            dll_name = os.path.basename(dll_path).lower()
            if "cuda" in dll_name and "_" in dll_name:
                parts = dll_name.split("_")
                if len(parts) > 1:
                    try:
                        ver_num = parts[-1].split(".")[0]
                        if ver_num.isdigit() and len(ver_num) >= 2:
                            major = ver_num[0:2] if len(ver_num) >= 3 else ver_num[0:1]
                            minor = ver_num[2:3] if len(ver_num) >= 3 else "0"
                            return f"{int(major)}.{int(minor)}", []
                    except (ValueError, IndexError):
                        pass
                        
        return "unknown", []
    except ImportError:
        return "not_installed", []

def create_launcher_scripts(dist_path, app_name):
    """Creates the launcher batch files for the application."""
    # Launcher for standard execution
    launcher_path = os.path.join(dist_path, f"Launch {app_name}.bat")
    with open(launcher_path, "w") as f:
        f.write(f'@echo off\n')
        f.write(f'echo Starting {app_name}...\n')
        f.write(f'cd /d "%~dp0{app_name}"\n')
        f.write(f'echo Launching from: %cd%\n')
        f.write(f'start "" "{app_name}.exe"\n')
        f.write(f'echo.\n')
        f.write(f'echo Application launched. The console can be closed.\n')

    print(f"Created launcher: {launcher_path}")

    # Launcher with CUDA path setup and console kept open for debugging
    cuda_launcher_path = os.path.join(dist_path, f"Launch {app_name} with CUDA.bat")
    with open(cuda_launcher_path, "w") as f:
        f.write(f'@echo off\n')
        f.write(f'title {app_name} - CUDA Console\n')
        f.write(f'echo Setting up CUDA environment for {app_name}...\n\n')

        # Add all known CUDA versions to the PATH
        for ver in [f"{i}.{j}" for i in range(11, 13) for j in range(10)]:
            f.write(f'if exist "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{ver}" (\n')
            f.write(f'    echo Found CUDA {ver} - Adding to PATH\n')
            f.write(f'    set "PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{ver}\\bin;%PATH%"\n')
            f.write(f'    set "CUDA_PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{ver}"\n')
            f.write(f'    goto :launch\n')
            f.write(f')\n\n')

        f.write(f':launch\n')
        f.write(f'cd /d "%~dp0{app_name}"\n')
        f.write(f'echo Current directory: %cd%\n')
        f.write(f'echo Launching with PATH: %PATH%\n\n')
        f.write(f'"{app_name}.exe"\n')
        f.write(f'echo.\n')
        f.write(f'echo Application finished. Press any key to exit.\n')
        f.write(f'pause > nul\n')

    print(f"Created CUDA launcher: {cuda_launcher_path}")


def prepare_llama_cpp():
    """Checks for llama-cpp-python and returns its paths."""
    try:
        import llama_cpp
        has_llama_cpp = True
        llamacpp_path = os.path.dirname(llama_cpp.__file__)
        llamacpp_lib_path = os.path.join(llamacpp_path, "lib")
        if not os.path.exists(llamacpp_lib_path):
            llamacpp_lib_path = None
        print("Found llama-cpp-python.")
        return has_llama_cpp, llamacpp_path, llamacpp_lib_path
    except ImportError:
        print("llama-cpp-python not found.")
        return False, None, None


def main():
    # Check for CUDA dependencies
    cuda_versions = find_cuda_dlls()
    print("\n=== CUDA DETECTION RESULTS ===")
    if cuda_versions:
        print("Found CUDA DLLs for these versions:")
        for version, dlls in cuda_versions.items():
            print(f"- CUDA {version}: {len(dlls)} DLLs")
            for i, (dll_name, _) in enumerate(dlls[:3]):  # Show only first 3
                print(f"  - {dll_name}")
            if len(dlls) > 3:
                print(f"  - ... and {len(dlls) - 3} more")
    else:
        print("No CUDA DLLs found in PATH. The packaged app will not have CUDA support.")

    # Check llama-cpp-python CUDA version
    llama_cpp_cuda_ver, required_dlls = detect_llama_cpp_cuda_version()
    print("\n=== LLAMA-CPP-PYTHON CUDA VERSION ===")
    if llama_cpp_cuda_ver == "unknown":
        print("Could not determine CUDA version for llama-cpp-python")
    elif llama_cpp_cuda_ver == "not_installed":
        print("llama-cpp-python is not installed")
    else:
        print(f"llama-cpp-python appears to require CUDA {llama_cpp_cuda_ver}")
        if required_dlls:
            print(f"Required DLLs: {', '.join(required_dlls)}")
        
        # Check if this CUDA version is installed
        if llama_cpp_cuda_ver in cuda_versions:
            print(f"✅ CUDA {llama_cpp_cuda_ver} is installed and available")
        else:
            print(f"⚠️ CUDA {llama_cpp_cuda_ver} is NOT installed! The packaged app may fail to load models.")
            print(f"   Consider installing CUDA {llama_cpp_cuda_ver} or rebuilding llama-cpp-python for CUDA {list(cuda_versions.keys())[0]}")

    print("\n=== CONTINUING WITH BUILD ===")

    # Clean previous builds
    if os.path.exists(os.path.join(PROJECT_ROOT, "build")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "build"))
    if os.path.exists(os.path.join(PROJECT_ROOT, "dist")):
        shutil.rmtree(os.path.join(PROJECT_ROOT, "dist"))

    # Create necessary directories that should be included
    for dir_name in ["docs", "img", "logs", "models"]:
        dir_path = os.path.join(PROJECT_ROOT, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    # Create a dummy settings file if it doesn't exist
    settings_path = os.path.join(PROJECT_ROOT, "settings.json")
    if not os.path.exists(settings_path):
        with open(settings_path, "w") as f:
            f.write('{"model": "qwen3:8b", "max_tokens": 8192, "custom_system_prompt": "", "timeout": 0}')
        print(f"Created default settings file")

    # Set icon path - the icon is in the same directory as this script (src folder)
    icon_path = os.path.join(SRC_DIR, "ladbon_ai.ico")
    if os.path.exists(icon_path):
        print(f"Using custom icon: {icon_path}")
    else:
        print("Warning: Custom icon 'ladbon_ai.ico' not found in src directory")

    # Check if llama-cpp-python is installed and prepare it
    has_llama_cpp, llamacpp_path, llamacpp_lib_path = prepare_llama_cpp()

    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Build Ladbon AI Desktop application')
    parser.add_argument('--onefile', action='store_true', help='Build as a single executable file')
    args = parser.parse_args()

    # Build the executable
    pyinstaller_args = [
        'gui_app.py',  # Your main entry point file
        f'--name={APP_NAME}',
        '--onefile' if args.onefile else '--onedir',  # Use onefile if specified, otherwise onedir
        '--windowed',
        f'--runtime-hook={os.path.join(SRC_DIR, "cuda_path_hook.py")}',  # Use the CUDA path hook
        f'--runtime-hook={os.path.join(SRC_DIR, "dll_load_hook.py")}',   # Also keep the DLL load hook
        # Use our enhanced hook script for better CUDA support - must be in current directory
        '--additional-hooks-dir=.',  # Look for hook files including hook-llama_cpp_enhanced.py
        '--copy-metadata=llama_cpp_python',  # Include llama_cpp metadata
        '--hidden-import=PyQt5.QtPrintSupport',  # Required for QTextEdit
        '--hidden-import=utils.ollama_client',
        '--hidden-import=utils.llamacpp_client',  # New llamacpp client
        '--hidden-import=utils.dll_loader',  # Important! DLL loader utility for llama-cpp
        '--hidden-import=utils.huggingface_manager',  # New HuggingFace manager
        '--hidden-import=utils.hf_auth',  # HuggingFace authentication utility
        '--hidden-import=utils.model_manager',  # New model manager
        '--hidden-import=utils.sanitycheck',  # New sanity check utility
        '--hidden-import=utils.data_paths',  # Include our data paths utility
        '--hidden-import=cli.doc_handler',
        '--hidden-import=cli.img_handler',
        '--hidden-import=cli.web_search',
        '--hidden-import=cli.rag',  # Include RAG module
        '--hidden-import=utils.logger',
        '--hidden-import=utils.numpy_init_fix',  # Add our NumPy fix module
        '--hidden-import=importlib',
        '--hidden-import=ctypes',
        '--hidden-import=logging',
        '--hidden-import=numpy',  # Explicitly include NumPy to ensure it's bundled correctly
    ]

    # Add icon to PyInstaller arguments if available and exists
    if icon_path and os.path.exists(icon_path):
        print(f"Adding icon to PyInstaller arguments: {icon_path}")
        pyinstaller_args.append(f'--icon={icon_path}')

    # Add manifest file for better Windows DLL handling
    manifest_path = os.path.join(SRC_DIR, "ladbon_ai_desktop.manifest")
    if os.path.exists(manifest_path):
        print(f"Using manifest file: {manifest_path}")
        pyinstaller_args.append(f'--manifest={manifest_path}')
    # Add llama-cpp-python related imports if available
    if has_llama_cpp:
        pyinstaller_args.extend([
            '--hidden-import=llama_cpp',
            '--hidden-import=llama_cpp.llama_cpp',
            '--hidden-import=llama_cpp._ctypes_extensions',
            '--hidden-import=llama_cpp.lib_path',  # Our auto-generated helper
            '--hidden-import=utils.dll_loader',    # Our new DLL loader utility
            '--hidden-import=psutil',  # Required by llamacpp_client
            '--hidden-import=huggingface_hub',  # Required for model download
            '--hidden-import=tqdm',  # Required for progress bars
        ])
        
        # Add llama_cpp directory and all its contents
        if llamacpp_path:
            # Add entire llama_cpp package as data
            print(f"Adding llama_cpp package from {llamacpp_path}")
            pyinstaller_args.append(f'--add-data={llamacpp_path}{os.pathsep}llama_cpp')
            
            # Add llama_cpp/lib directory and all its contents if it exists
            if llamacpp_lib_path and os.path.exists(llamacpp_lib_path):
                print(f"Adding llama_cpp lib directory from {llamacpp_lib_path}")
                
                # Add the lib directory and its contents as binary files
                # This is critical to ensure the library loads correctly
                pyinstaller_args.append(f'--add-binary={llamacpp_lib_path}{os.pathsep}llama_cpp{os.path.sep}lib')
                
                # We no longer copy DLLs to the root directory
                # This was causing conflicts with the properly bundled DLLs
                # The DLLs should only be in the llama_cpp/lib directory
                if platform.system() == "Windows":
                    print("NOTE: DLLs will only be placed in llama_cpp/lib directory for proper loading")
                
                # Also add the empty directory structure (important for PyInstaller to create the directory)
                #pyinstaller_args.append(f'--add-data={llamacpp_lib_path}{os.pathsep};llama_cpp{os.path.sep}lib')
            
        # We no longer copy all DLLs from site-packages
        # PyInstaller should handle dependencies automatically
        # Adding them all can cause conflicts with the correctly bundled DLLs
        if platform.system() == "Windows":
            print("Relying on PyInstaller to handle DLL dependencies. Forcing inclusion of numpy and llama_cpp.")
            try:
                import numpy
                numpy_path = os.path.dirname(numpy.__file__)
                pyinstaller_args.append(f'--add-data={numpy_path}{os.pathsep}numpy')
                print(f"Force-including numpy from {numpy_path}")
            except ImportError:
                print("Numpy not found, skipping forced inclusion.")

    # Check for platform-specific requirements
    if platform.system() == "Windows":
        pyinstaller_args.append('--hidden-import=win32process')

    # Add additional hooks path for custom hooks
    pyinstaller_args.append(f'--additional-hooks-dir={SRC_DIR}')

    # Runtime hooks already specified in main arguments list
    # No need to add them again here

    # Always add the clean flag at the end
    pyinstaller_args.append('--clean')

    # Run PyInstaller with our arguments
    print("Building the executable... (this may take a few minutes)")
    PyInstaller.__main__.run(pyinstaller_args)

    print(f"\nBuild completed! The executable is in the 'dist' folder.")

    # --- Post-build tasks ---
    dist_path = os.path.join(PROJECT_ROOT, "dist")
    app_name = APP_NAME

    # Create launcher scripts
    create_launcher_scripts(dist_path, app_name)

    # Print appropriate messages about model support
    if has_llama_cpp:
        print(f"Local LLM support is included via llama-cpp-python.")
        print(f"Copy your GGUF models to the models folder in the application directory.")
        
        # Show CUDA compatibility warnings if needed
        if llama_cpp_cuda_ver != "unknown" and llama_cpp_cuda_ver != "not_installed":
            if llama_cpp_cuda_ver in cuda_versions:
                print(f"\n✅ CUDA {llama_cpp_cuda_ver} is installed and available for llama-cpp-python.")
            else:
                print(f"\n⚠️ WARNING: llama-cpp-python was built for CUDA {llama_cpp_cuda_ver} but this version wasn't found on your system.")
                available_versions = list(cuda_versions.keys())
                if available_versions:
                    print(f"   You have CUDA {', '.join(available_versions)} installed.")
                    print(f"   To use GPU acceleration, you need to either:")
                    print(f"   1. Install CUDA {llama_cpp_cuda_ver} from the NVIDIA website, or")
                    print(f"   2. Rebuild llama-cpp-python for your existing CUDA version {available_versions[0]}")
                else:
                    print(f"   No CUDA installation was detected.")
                    print(f"   To use GPU acceleration, install CUDA {llama_cpp_cuda_ver} from the NVIDIA website.")
                print(f"   The packaged app will still work but may fall back to CPU-only mode.")
    else:
        print(f"WARNING: Local LLM support via llama-cpp-python was NOT included in this build.")
        print(f"Run install_llamacpp.py and rebuild if you want local model support.")

    print(f"\nNOTE: You can also install Ollama separately for additional model options: https://ollama.com/download")
    print(f"Remember to copy any document or image files you need to the 'docs' and 'img' folders in the application directory.")

if __name__ == "__main__":
    main()