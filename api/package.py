import PyInstaller.__main__
import os
import shutil

# Define the application name
APP_NAME = "LocalAI"

# Clean previous builds
if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists("dist"):
    shutil.rmtree("dist")

# Build the application
PyInstaller.__main__.run([
    'gui_app.py',  # This is correct if run from the project root
    '--name=%s' % APP_NAME,
    '--onefile',
    '--windowed',
    '--icon=app_icon.ico',  # Create/add an icon file
    '--add-data=models;models',  # Include any needed directories
])

print("Build completed!")
print(f"Executable is at: dist/{APP_NAME}.exe")