import os
import sys
import subprocess
import platform

def main():
    """CPU-only installation of llama-cpp-python to avoid GPU-related crashes"""
    print("Installing CPU-only version of llama-cpp-python")
    print("This will fix 'access violation reading' errors")
    
    # First uninstall any existing version
    print("\n1. Removing any existing installation...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
        print("   ✓ Successfully removed existing installation")
    except:
        print("   ✓ No existing installation found")
    
    # Set environment variables to force CPU-only build
    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=OFF -DLLAMA_METAL=OFF"
    env["FORCE_CMAKE"] = "1"
    
    print("\n2. Installing CPU-only version...")
    print("   (This may take a few minutes as it builds from source)")
    
    try:
        # Install a different version that doesn't have the numa parameter issue
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "llama-cpp-python==0.2.23"],
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        print("   ✓ Installation complete!")
        print("\n3. Testing installation...")
        
        # Try to import and check if initialization works
        test_script = """
import llama_cpp
print(f"   ✓ Successfully imported llama-cpp-python version {llama_cpp.__version__}")
try:
    # This version shouldn't need the numa parameter
    llama_cpp.llama_backend_init()
    print("   ✓ Backend initialized successfully")
    print("\\n✅ Installation successful! You should no longer see access violation errors.")
    print("   Restart your application to use the new installation.")
except Exception as e:
    print(f"   ✗ Backend initialization failed: {e}")
    print("   The installation completed but there might still be issues.")
"""
        subprocess.run([sys.executable, "-c", test_script], check=False)
        
    except subprocess.CalledProcessError as e:
        print("\n❌ Installation failed:")
        print(e.stdout)
        print("\nPlease make sure you have the required build tools installed:")
        print("- Visual Studio Build Tools with C++ workload (on Windows)")
        print("- CMake")
        print("- A C++ compiler")
        
    print("\nIf you want to try again with a different version, run:")
    print("pip install llama-cpp-python==0.1.78 --force-reinstall")

if __name__ == "__main__":
    main()