"""
Test script for llama-cpp-python DLL loading
This script tests if llama-cpp-python is properly loaded with CUDA support
"""

import os
import sys
import platform
import traceback
from pathlib import Path

print(f"Python version: {platform.python_version()}")
print(f"Platform: {platform.platform()}")
print(f"System architecture: {platform.architecture()}")

# Add DLL directories if running from a bundled app
if getattr(sys, 'frozen', False):
    # When running as bundled app, PyInstaller provides _MEIPASS attribute
    base_dir = getattr(sys, '_MEIPASS', None)
    if base_dir is None:
        base_dir = os.path.dirname(sys.executable)
    
    print(f"Running as bundled app from: {base_dir}")
    
    # Look for DLL directories
    dll_dirs = [
        os.path.join(base_dir, "_internal", "llama_cpp", "lib"),
        os.path.join(base_dir, "llama_cpp", "lib"),
        os.path.join(base_dir, "_internal", "cuda_dlls"),
        os.path.join(base_dir, "cuda_dlls")
    ]
    
    for dll_dir in dll_dirs:
        if os.path.exists(dll_dir):
            # Add directory to DLL search path
            try:
                os.add_dll_directory(dll_dir)
                print(f"Added DLL directory: {dll_dir}")
            except Exception as e:
                print(f"Error adding DLL directory {dll_dir}: {e}")
            
            # Update PATH environment variable
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            print(f"Added to PATH: {dll_dir}")
            
            # List DLLs in this directory
            dlls = list(Path(dll_dir).glob("*.dll"))
            print(f"Found {len(dlls)} DLLs in {dll_dir}")
            for dll in dlls[:5]:
                print(f"  - {dll.name}")
            if len(dlls) > 5:
                print(f"  - ... and {len(dlls) - 5} more")
else:
    print("Running in development mode (not bundled)")

# Check for CUDA installations
cuda_paths = []
if platform.system() == "Windows":
    base_dirs = [
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "C:/Program Files/NVIDIA Corporation"
    ]
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            print(f"Found CUDA base directory: {base_dir}")
            # Look for version subdirectories
            for item in os.listdir(base_dir):
                version_dir = os.path.join(base_dir, item)
                if os.path.isdir(version_dir) and ("v" in item or "." in item):
                    bin_dir = os.path.join(version_dir, "bin")
                    if os.path.exists(bin_dir):
                        cuda_paths.append(bin_dir)
                        print(f"Found CUDA bin directory: {bin_dir}")
                        
                        # Add to PATH and DLL search path
                        try:
                            os.add_dll_directory(bin_dir)
                        except Exception as e:
                            print(f"Error adding DLL directory {bin_dir}: {e}")
                        
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                        print(f"Added to PATH: {bin_dir}")

# Try to import llama_cpp
print("\nTrying to import llama_cpp...")
try:
    import llama_cpp
    print(f"Success! llama_cpp version: {getattr(llama_cpp, '__version__', 'unknown')}")
    print(f"llama_cpp module location: {llama_cpp.__file__}")
    
    # Check for CUDA capabilities
    print("\nChecking for CUDA capabilities...")
    if hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib"):
        lib = llama_cpp.llama_cpp._lib
        cuda_functions = [attr for attr in dir(lib) if "cuda" in attr.lower() or "gpu" in attr.lower()]
        if cuda_functions:
            print(f"Found {len(cuda_functions)} CUDA-related functions:")
            for func in cuda_functions[:5]:
                print(f"  - {func}")
            if len(cuda_functions) > 5:
                print(f"  - ... and {len(cuda_functions) - 5} more")
        else:
            print("No CUDA functions found in llama_cpp")
    
    # Try to initialize backend
    print("\nTrying to initialize backend...")
    if hasattr(llama_cpp, "llama_backend_init"):
        try:
            import inspect
            sig = inspect.signature(llama_cpp.llama_backend_init)
            params = list(sig.parameters.keys())
            print(f"llama_backend_init takes parameters: {params}")
            
            print("Trying initialization...")
            # Define initialization success flag
            init_success = False
            
            # Try different approaches for different llama_cpp_python versions
            try:
                # First try without parameters - this works in most versions
                print("Calling without parameters...")
                llama_cpp.llama_backend_init()
                init_success = True
                print("Successfully initialized with no parameters")
            except Exception as e1:
                print(f"Error with no parameters: {e1}")
                
                # Try with parameters if available in the signature
                if len(params) > 0:
                    try:
                        # Try using kwargs with first parameter
                        param_name = params[0]
                        kwargs = {param_name: False}
                        llama_cpp.llama_backend_init(**kwargs)
                        init_success = True
                        print(f"Successfully initialized with kwargs: {param_name}=False")
                    except Exception as e2:
                        print(f"Error initializing with kwargs: {e2}")
                
                # As a last resort, try using the C library directly
                if not init_success:
                    try:
                        if hasattr(llama_cpp, "llama_cpp") and hasattr(llama_cpp.llama_cpp, "_lib"):
                            print("Trying direct C function call")
                            lib_func = getattr(llama_cpp.llama_cpp._lib, "llama_backend_init")
                            # Call without arguments first
                            try:
                                lib_func()
                                init_success = True
                                print("Successfully initialized with direct C call (no args)")
                            except Exception:
                                # Try with a single False argument
                                try:
                                    lib_func(False)
                                    init_success = True
                                    print("Successfully initialized with direct C call (False arg)")
                                except Exception as e3:
                                    print(f"Direct C function call failed: {e3}")
                        else:
                            print("No direct C function access available")
                    except Exception as e4:
                        print(f"All initialization attempts failed: {e4}")
            
            if init_success:
                print("Backend initialization successful!")
            
            # Try to load a model if specified
            model_path = os.environ.get("LLAMA_TEST_MODEL")
            if model_path and os.path.exists(model_path):
                print(f"\nTrying to load model: {model_path}")
                try:
                    # Try with GPU
                    print("Loading with n_gpu_layers=1...")
                    model = llama_cpp.Llama(
                        model_path=model_path,
                        n_gpu_layers=1,
                        verbose=True
                    )
                    print("Successfully loaded model with GPU!")
                    
                    # Try a simple generation
                    print("\nTrying a simple generation...")
                    output = model("Q: What is the capital of France? A:", max_tokens=5)
                    print(f"Generation result: {output}")
                    
                except Exception as model_err:
                    print(f"Error loading model with GPU: {model_err}")
                    # Try with CPU as fallback
                    try:
                        print("\nTrying with CPU fallback...")
                        model = llama_cpp.Llama(
                            model_path=model_path,
                            n_gpu_layers=0
                        )
                        print("Successfully loaded model with CPU only")
                    except Exception as cpu_err:
                        print(f"Error loading model with CPU: {cpu_err}")
            else:
                print("\nNo test model specified in LLAMA_TEST_MODEL environment variable")
                print("Skipping model loading test")
        
        except Exception as init_err:
            print(f"Error initializing backend: {init_err}")
            print(traceback.format_exc())
    else:
        print("llama_backend_init not found in module")
        
except ImportError as e:
    print(f"Failed to import llama_cpp: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    print(traceback.format_exc())
