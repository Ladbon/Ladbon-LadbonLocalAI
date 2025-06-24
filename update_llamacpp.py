import os
import sys
import subprocess
import platform

def main():
    """Helper script to update llama-cpp-python to support newer architectures"""
    print("Updating llama-cpp-python to support Gemma models...")
    
    # First uninstall any existing version
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
        print("Successfully uninstalled previous version")
    except:
        print("No previous version found or error uninstalling")
    
    # Install newer version for Gemma support
    print("\nInstalling newer version for Gemma support...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--no-cache-dir", "--force-reinstall",
            "llama-cpp-python>=0.3.0"  # Version 0.3.0+ supports Gemma architecture
        ])
        print("\nSuccessfully installed llama-cpp-python with Gemma support")
        print("\nRestart the application to use the new installation")
    except Exception as e:
        print(f"\nError installing updated version: {str(e)}")
        
if __name__ == "__main__":
    main()