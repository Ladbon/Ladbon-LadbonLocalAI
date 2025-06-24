import os
import shutil
import sys
import subprocess
import PyInstaller.__main__
import platform

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
PROJECT_ROOT = os.path.dirname(SRC_DIR)
APP_NAME = "Ladbon AI Desktop"

# Move to the correct directory
os.chdir(SRC_DIR)

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

# Skip icon creation since it's causing errors
print("Skipping icon creation - will use default icon.")
icon_path = None  # No custom icon

# Check if llama-cpp-python is installed
try:
    import llama_cpp
    has_llama_cpp = True
    print("llama-cpp-python is installed")
except ImportError:
    has_llama_cpp = False
    print("WARNING: llama-cpp-python is not installed.")
    print("Local model inference will not be available.")
    print("Run install_llamacpp.py first if you want local model support.")

# Build the executable
pyinstaller_args = [
    'gui_app.py',  # Your main entry point file
    f'--name={APP_NAME}',
    '--onefile',
    '--windowed',
    '--add-data={}{}docs;docs'.format(PROJECT_ROOT, os.sep),
    '--add-data={}{}img;img'.format(PROJECT_ROOT, os.sep),
    '--add-data={}{}logs;logs'.format(PROJECT_ROOT, os.sep),
    '--add-data={}{}models;models'.format(PROJECT_ROOT, os.sep),  # Include models directory
    '--add-data={}{}settings.json;.'.format(PROJECT_ROOT, os.sep),
    '--hidden-import=PyQt5.QtPrintSupport',  # Required for QTextEdit
    '--hidden-import=utils.ollama_client',
    '--hidden-import=utils.llamacpp_client',  # New llamacpp client
    '--hidden-import=utils.huggingface_manager',  # New HuggingFace manager
    '--hidden-import=utils.model_manager',  # New model manager
    '--hidden-import=utils.sanitycheck',  # New sanity check utility
    '--hidden-import=cli.doc_handler',
    '--hidden-import=cli.img_handler',
    '--hidden-import=cli.web_search',
    '--hidden-import=cli.rag',  # Include RAG module
    '--hidden-import=utils.logger',
    '--hidden-import=importlib',
    '--hidden-import=ctypes',
]

# Add llama-cpp-python related imports if available
if has_llama_cpp:
    pyinstaller_args.extend([
        '--hidden-import=llama_cpp',
        '--hidden-import=llama_cpp.llama_cpp',
        '--hidden-import=psutil',  # Required by llamacpp_client
        '--hidden-import=huggingface_hub',  # Required for model download
        '--hidden-import=tqdm',  # Required for progress bars
    ])

# Check for platform-specific requirements
if platform.system() == "Windows":
    pyinstaller_args.append('--hidden-import=win32process')

# Always add the clean flag at the end
pyinstaller_args.append('--clean')

# Run PyInstaller with our arguments
print("Building the executable... (this may take a few minutes)")
PyInstaller.__main__.run(pyinstaller_args)

print(f"\nBuild completed! The executable is in the 'dist' folder.")

# Print appropriate messages about model support
if has_llama_cpp:
    print(f"Local LLM support is included via llama-cpp-python.")
    print(f"Copy your GGUF models to the 'models' folder in the same directory as the .exe")
else:
    print(f"WARNING: Local LLM support via llama-cpp-python was NOT included in this build.")
    print(f"Run install_llamacpp.py and rebuild if you want local model support.")

print(f"NOTE: You can also install Ollama separately for additional model options: https://ollama.com/download")
print(f"Remember to copy any document or image files you need to the 'docs' and 'img' folders in the same directory as the .exe")