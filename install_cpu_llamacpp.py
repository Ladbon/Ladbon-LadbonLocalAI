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
    
    # For the pre-built CPU-only wheel
    print("\n2. Installing CPU-only version (0.3.9+cpu)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "llama-cpp-python==0.3.9+cpu"
        ])
        print("   ✓ Successfully installed CPU-only version")
    except subprocess.CalledProcessError:
        print("   ✗ Failed to install pre-built CPU wheel")
        
        # Fall back to building from source with CPU only
        print("\n   Falling back to building from source (CPU only)...")
        
        # Set environment variables to force CPU-only build
        env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=OFF"
        env["FORCE_CMAKE"] = "1"
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python==0.3.9", "--no-binary", "llama-cpp-python"
            ], env=env)
            print("   ✓ Successfully built CPU-only version from source")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed to build from source: {e}")
            print("Installation failed. Please check your build environment.")
            return False
    
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

if __name__ == "__main__":
    main()