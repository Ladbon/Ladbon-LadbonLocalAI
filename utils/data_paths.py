"""
Utility module for managing application data paths.
This ensures all user-writable data is stored in the appropriate location
(AppData on Windows, home directory on other platforms).
"""
import os
import sys
import platform
import shutil
from pathlib import Path

APP_NAME = "Ladbon AI Desktop"

def get_app_data_dir():
    """
    Returns the application data directory.
    For a packaged app, this is the directory containing the executable.
    For a script, it's the project's src directory.
    """
    if getattr(sys, 'frozen', False):
        # The application is running as a bundled executable (e.g., PyInstaller)
        # Use the directory of the executable as the base path
        app_data_dir = os.path.dirname(sys.executable)
    else:
        # The application is running as a standard Python script
        # Use the project's root directory (assuming this file is in src/utils)
        app_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # The 'logs', 'models', etc. directories will be created inside this path
    return app_data_dir

def get_data_path(folder_name):
    """
    Get the path to a specific data folder within the application data directory.
    Creates the directory if it doesn't exist.
    
    Args:
        folder_name: The name of the folder (e.g., 'docs', 'img', 'logs', 'models')
        
    Returns:
        The absolute path to the requested folder
    """
    path = os.path.join(get_app_data_dir(), folder_name)
    os.makedirs(path, exist_ok=True)
    return path

def get_settings_path():
    """Get the absolute path to the settings.json file"""
    return os.path.join(get_app_data_dir(), "settings.json")

def get_docs_dir():
    """Get the absolute path to the docs directory"""
    return get_data_path('docs')

def get_img_dir():
    """Get the absolute path to the img directory"""
    return get_data_path('img')

def get_logs_dir():
    """Get the absolute path to the logs directory"""
    return get_data_path('logs')

def get_models_dir():
    """Get the absolute path to the models directory"""
    return get_data_path('models')

def migrate_data(src_folder, dest_folder):
    """
    Migrate data from one folder to another if needed.
    This is useful for migrating from old installations.
    
    Args:
        src_folder: Source folder path
        dest_folder: Destination folder path
    """
    if not os.path.exists(src_folder) or not os.path.isdir(src_folder):
        return False
        
    # Create destination if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Copy all files from source to destination
    for item in os.listdir(src_folder):
        s = os.path.join(src_folder, item)
        d = os.path.join(dest_folder, item)
        
        if os.path.isfile(s):
            # Don't overwrite existing files
            if not os.path.exists(d):
                shutil.copy2(s, d)
        elif os.path.isdir(s):
            # Recursively copy subdirectories
            if not os.path.exists(d):
                shutil.copytree(s, d)
    
    return True

def first_run_migration():
    """
    Check if this is the first run after installation and
    migrate any existing data from the installation directory
    to the user data directory.
    """
    # Only run migration if we're in a packaged app
    if not (getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')):
        return
        
    app_dir = get_app_data_dir()
    # Use a marker file to track if we've already migrated
    migration_marker = os.path.join(app_dir, '.migration_complete')
    
    if os.path.exists(migration_marker):
        return  # Migration already done
        
    # Get the installation directory (where the .exe is)
    if platform.system() == 'Windows':
        install_dir = os.path.dirname(sys.executable)
        
        # Migrate data from installation directory to user data directory
        for folder in ['docs', 'img', 'models']:
            src = os.path.join(install_dir, folder)
            dest = os.path.join(app_dir, folder)
            migrate_data(src, dest)
            
        # Migrate settings.json
        src_settings = os.path.join(install_dir, 'settings.json')
        dest_settings = os.path.join(app_dir, 'settings.json')
        if os.path.exists(src_settings) and not os.path.exists(dest_settings):
            shutil.copy2(src_settings, dest_settings)
    
    # Create marker file to indicate migration is complete
    with open(migration_marker, 'w') as f:
        f.write('Migration completed')
