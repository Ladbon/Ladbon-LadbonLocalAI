import os
import shutil
import sys
import subprocess
import PyInstaller.__main__

# Check if PyInstaller is installed
try:
    import PyInstaller
except ImportError:
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

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
for dir_name in ["docs", "img", "logs"]:
    dir_path = os.path.join(PROJECT_ROOT, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# Create a dummy settings file if it doesn't exist
settings_path = os.path.join(PROJECT_ROOT, "settings.json")
if not os.path.exists(settings_path):
    with open(settings_path, "w") as f:
        f.write('{"model": "qwen3:8b", "max_tokens": 8192}')
    print(f"Created default settings file")

# Skip icon creation since it's causing errors
print("Skipping icon creation - will use default icon.")
icon_path = None  # No custom icon

# Build the executable
pyinstaller_args = [
    'gui_app.py',  # Your main entry point file
    f'--name={APP_NAME}',
    '--onefile',
    '--windowed',
    '--add-data={}{}docs;docs'.format(PROJECT_ROOT, os.sep),
    '--add-data={}{}img;img'.format(PROJECT_ROOT, os.sep),
    '--add-data={}{}logs;logs'.format(PROJECT_ROOT, os.sep),
    '--add-data={}{}settings.json;.'.format(PROJECT_ROOT, os.sep),
    '--hidden-import=PyQt5.QtPrintSupport',  # Required for QTextEdit
    '--hidden-import=utils.ollama_client',
    '--hidden-import=cli.doc_handler',
    '--hidden-import=cli.img_handler',
    '--hidden-import=cli.web_search',
    '--hidden-import=utils.logger',
    '--clean'
]

# Run PyInstaller with our arguments
print("Building the executable... (this may take a few minutes)")
PyInstaller.__main__.run(pyinstaller_args)

print(f"\nBuild completed! The executable is in the 'dist' folder.")
print(f"NOTE: You still need to install Ollama separately: https://ollama.com/download")
print(f"Remember to copy any document or image files you need to the 'docs' and 'img' folders in the same directory as the .exe")