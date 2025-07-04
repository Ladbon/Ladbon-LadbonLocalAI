import os, sys

# Handle PyInstaller bundled app and regular execution
if getattr(sys, 'frozen', False):
    # Get base directory from PyInstaller's special attribute or fallback to executable location
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    for subdir in ["_internal/llama_cpp/lib", "llama_cpp/lib"]:
        dll_dir = os.path.join(base_dir, subdir)
        if os.path.exists(dll_dir):
            # Add directory to DLL search path and update PATH environment variable
            os.add_dll_directory(dll_dir)
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ["PATH"]
            print(f"Added DLL directory: {dll_dir}")

try:
    import llama_cpp
    print("llama_cpp version:", llama_cpp.__version__)
    
    # Check if llama_backend_init needs arguments
    import inspect
    
    try:
        sig = inspect.signature(llama_cpp.llama_backend_init)
        params = list(sig.parameters.keys())
        print(f"llama_backend_init takes parameters: {params}")
        
        # Try to initialize without arguments - this works for newer versions
        # Define initialization success flag
        init_success = False
        
        # Try initialization approaches based on the signature
        if len(params) == 0:
            # No parameters in the signature
            try:
                # Call with no parameters
                llama_cpp.llama_backend_init()
                print("Successfully initialized with no parameters")
                init_success = True
            except Exception as e1:
                print(f"Error initializing with no parameters: {e1}")
        else:
            # Has parameters in the signature
            try:
                # Try using kwargs with first parameter
                param_name = params[0]
                kwargs = {param_name: False}
                llama_cpp.llama_backend_init(**kwargs)
                print(f"Successfully initialized with kwargs: {param_name}=False")
                init_success = True
            except Exception as e2:
                print(f"Error initializing with kwargs: {e2}")
                
                # Try using dynamic approach
                try:
                    backend_init_func = getattr(llama_cpp, "llama_backend_init")
                    # Use reflection to safely call the function
                    backend_init_func.__call__(False)
                    print("Successfully initialized with __call__ method")
                    init_success = True
                except Exception as e3:
                    print(f"All initialization attempts failed: {e3}")
                    print("Continuing without initialization")
            
        print("llama_backend_init succeeded!")
    except Exception as init_err:
        print(f"Error initializing backend: {init_err}")
        
except Exception as e:
    print("ERROR:", e)
