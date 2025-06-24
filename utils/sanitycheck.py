import importlib
import sys
import pathlib
import ctypes

# First check if we need to patch the backend
def ensure_patched_backend():
    try:
        import llama_cpp
        
        # Check if it's already patched by gui_app.py
        if hasattr(llama_cpp.llama_backend_init, '__name__') and \
           llama_cpp.llama_backend_init.__name__ == '_patched_backend_init':
            print("Backend already patched by gui_app.py")
            return True
            
        # If not patched yet, apply our own patch
        if hasattr(llama_cpp, '_lib') and hasattr(llama_cpp.llama_cpp._lib, 'llama_backend_init'):
            original_argtypes = getattr(llama_cpp.llama_cpp._lib.llama_backend_init, 'argtypes', None)
            
            # Set proper argument types if needed
            if original_argtypes is None or original_argtypes != [ctypes.c_bool]:
                llama_cpp.llama_cpp._lib.llama_backend_init.argtypes = [ctypes.c_bool]
                llama_cpp.llama_cpp._lib.llama_backend_init.restype = None
                
                # Create and apply patched function
                def _patched_backend_init(numa=False):
                    return llama_cpp.llama_cpp._lib.llama_backend_init(ctypes.c_bool(numa))
                
                _patched_backend_init.__name__ = '_patched_backend_init'
                llama_cpp.llama_backend_init = _patched_backend_init
                print("Applied local backend patch in sanitycheck.py")
                
            return True
        else:
            print("Cannot patch backend: required attributes not found")
            return False
    except ImportError:
        print("Cannot import llama_cpp")
        return False
    except Exception as e:
        print(f"Error patching backend: {e}")
        return False

# Apply patch and run checks
if ensure_patched_backend():
    import llama_cpp
    
    print("Loaded from:", pathlib.Path(llama_cpp.__file__).resolve())
    print("_lib present:", hasattr(llama_cpp, "_lib"))
    
    # Now we can safely call the patched function
    llama_cpp.llama_backend_init()  # This will work regardless of version
    llama_cpp.llama_backend_free()
    
    print("Backend initialized and freed successfully – ready to load models.")
else:
    print("Failed to patch backend - model loading may fail.")