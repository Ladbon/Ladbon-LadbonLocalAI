import os
import sys
import subprocess
import platform
import argparse

def main():
    parser = argparse.ArgumentParser(description="Install LlamaCPP with appropriate settings")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only installation")
    parser.add_argument("--version", default="0.3.9", help="Version of llama-cpp-python to install")
    parser.add_argument("--force", action="store_true", help="Force reinstallation")
    args = parser.parse_args()

    print(f"Installing llama-cpp-python {args.version}...")
    
    # First uninstall current version if forced
    if args.force:
        print("\n1. Removing existing installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])
            print("   ✓ Successfully removed existing installation")
        except:
            print("   ✓ No existing installation found")
    
    # Set environment variables for compilation
    env = os.environ.copy()
    env["FORCE_CMAKE"] = "1"
    
    if args.cpu_only:
        print("\nUsing CPU-only mode (no GPU acceleration)")
        # For 0.3.x+ use DGGML_CUDA=OFF
        if float(args.version.split(".")[1]) >= 3:
            env["CMAKE_ARGS"] = "-DGGML_CUDA=OFF -DLLAMA_METAL=OFF"
        else: # For 0.2.x use LLAMA_CUBLAS=OFF
            env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=OFF -DLLAMA_METAL=OFF"
    else:
        # Try to detect GPU
        use_cuda = False
        try:
            if platform.system() == "Windows":
                output = subprocess.check_output("nvidia-smi", shell=True, stderr=subprocess.DEVNULL)
                use_cuda = True
                print("NVIDIA GPU detected, configuring for CUDA support")
        except:
            print("No NVIDIA GPU detected or nvidia-smi not available, using CPU only")
            
        # Set flags according to version
        if float(args.version.split(".")[1]) >= 3:
            env["CMAKE_ARGS"] = "-DGGML_CUDA=ON" if use_cuda else "-DGGML_CUDA=OFF"
        else:
            env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=ON" if use_cuda else "-DLLAMA_CUBLAS=OFF"
    
    # Run the installation command
    cmd = [
        sys.executable, "-m", "pip", "install", 
        f"llama-cpp-python=={args.version}",
        "--no-cache-dir", "--verbose"
    ]
    
    if args.force:
        cmd.append("--force-reinstall")
        
    print(f"\n2. Installing llama-cpp-python {args.version}...")
    print(f"Running: {' '.join(cmd)}")
    print(f"With CMAKE_ARGS: {env['CMAKE_ARGS']}")
    
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print("\nSuccessfully installed llama-cpp-python!")
        
        # Test importing
        print("\n3. Testing installation...")
        test_script = """
import llama_cpp
print(f"Successfully imported llama-cpp-python version {llama_cpp.__version__}")
try:
    llama_cpp.llama_backend_init()
    print("Backend initialized successfully")
    print("\\n✅ Installation successful!")
    print("Restart your application to use the new installation.")
except Exception as e:
    print(f"Backend initialization failed: {e}")
    print("The installation completed but there might still be issues.")
"""
        subprocess.run([sys.executable, "-c", test_script], check=False)
    else:
        print("\nInstallation failed with error code:", result.returncode)
        print("\nFalling back to known working version...")
        
        # Fall back to a known working version
        fallback_cmd = [
            sys.executable, "-m", "pip", "install",
            "llama-cpp-python==0.2.56",
            "--no-cache-dir"
        ]
        subprocess.run(fallback_cmd)

if __name__ == "__main__":
    main()