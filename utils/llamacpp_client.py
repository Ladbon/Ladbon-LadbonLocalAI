import os
import gc
import sys
import platform
import psutil
import importlib, ctypes # Keep for patch logic if moved here
import traceback

try:
    import llama_cpp
except ImportError:
    # This will be logged by _check_llamacpp_version if it's called first
    pass # Or log an immediate error if direct usage is expected before _check_llamacpp_version

from pathlib import Path
from utils.logger import setup_logger
from utils.data_paths import get_models_dir
from utils.dll_loader import ensure_dlls_loadable

logger = setup_logger('llamacpp_client')

# Try to ensure DLLs are properly loaded at import time
if platform.system() == "Windows":
    dll_load_success = ensure_dlls_loadable()
    if dll_load_success:
        logger.info("DLL loader pre-initialization successful")
    else:
        logger.warning("DLL loader pre-initialization failed - may have issues loading models")

class LlamaCppClient:
    def __init__(self, n_ctx=4096, n_gpu_layers=4): # Accept parameters
        logger.info(f"Initializing LlamaCppClient with n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")
        self.loaded_model = None
        self.model_path = None
        self.n_ctx = n_ctx  # Use passed value
        self.n_gpu_layers = n_gpu_layers # Use passed value
        self.model_metadata = {}
        self.llamacpp_version = "unknown" # Default before check
        self.is_v3_api = False # Default before check
        self._check_llamacpp_version()
        logger.info("LlamaCppClient initialized.")

    def update_config(self, n_ctx=None, n_gpu_layers=None):
        """Allows updating config after initialization, e.g., if settings change."""
        if n_ctx is not None:
            self.n_ctx = n_ctx
            logger.info(f"LlamaCppClient n_ctx updated to: {self.n_ctx}")
        if n_gpu_layers is not None:
            self.n_gpu_layers = n_gpu_layers
            logger.info(f"LlamaCppClient n_gpu_layers updated to: {self.n_gpu_layers}")
        # If a model is loaded, it might need to be reloaded for changes to take effect
        if self.loaded_model and (n_ctx is not None or n_gpu_layers is not None):
            logger.warning("n_ctx or n_gpu_layers changed. Model may need to be reloaded for changes to apply.")

    def _check_llamacpp_version(self):
        logger.debug("Checking llama-cpp-python version and capabilities...")
        try:
            # Ensure llama_cpp is imported for version check
            import llama_cpp as lcpp_module_for_version_check
            self.llamacpp_version = lcpp_module_for_version_check.__version__
            logger.info(f"Found llama-cpp-python version: {self.llamacpp_version}")
            
            try:
                major_str, minor_str, *_ = self.llamacpp_version.split('.')
                major = int(major_str)
                minor = int(minor_str)
                # Assuming 0.2.x is old API, 0.3.x+ is new API
                # This logic might need refinement based on actual API changes
                if major == 0 and minor >= 3: # Example threshold
                    self.is_v3_api = True 
                else:
                    self.is_v3_api = False
                logger.info(f"Determined API version usage: {'v3+ (newer)' if self.is_v3_api else 'v2- (older)'}")
                
                # Check for CUDA (this is a placeholder, actual check is more complex)
                # The true check is if llama.cpp was compiled with CUDA and if n_gpu_layers > 0
                # We can't easily check compilation flags from here without trying to load with n_gpu_layers.
                logger.info("CUDA support: To enable, ensure llama-cpp-python was built with CUDA and set n_gpu_layers > 0 in settings.")
                
            except ValueError:
                 logger.error(f"Could not parse llama-cpp-python version string: {self.llamacpp_version}")
                 self.is_v3_api = False 
            except Exception as inner_e:
                logger.exception(f"Error while checking API details post-version retrieval: {inner_e}")
                self.is_v3_api = False
                
        except ImportError:
            logger.error("llama-cpp-python is not installed. Integrated backend will not function.")
            self.llamacpp_version = "not installed"
            self.is_v3_api = False
        except Exception as e:
            logger.exception(f"Unexpected error checking llama-cpp-python version: {e}")
            self.llamacpp_version = "unknown"
            self.is_v3_api = False

    def _safe_init_backend(self):
        """
        Initialize llama-cpp backend safely, compatible with the patch in gui_app.py
        With enhanced error logging for better DLL debugging
        """
        logger.debug("Entering _safe_init_backend.")
        
        # Check system architecture first
        is_64bit = platform.architecture()[0] == '64bit'
        logger.info(f"System architecture: {'64-bit' if is_64bit else '32-bit'}")
        
        is_python_64bit = sys.maxsize > 2**32
        logger.info(f"Python interpreter: {'64-bit' if is_python_64bit else '32-bit'}")
        
        if not is_python_64bit:
            logger.error("CRITICAL ERROR: Running in 32-bit Python! llama-cpp-python requires 64-bit Python")
            return False
            
        # First, explicitly force the DLLs to be loaded
        if platform.system() == "Windows":
            try:
                from utils.dll_loader import ensure_dlls_loadable
                dll_load_result = ensure_dlls_loadable()
                logger.info(f"DLL pre-loading result: {'Success' if dll_load_result else 'Failed'}")
            except Exception as e:
                logger.error(f"Failed to pre-load DLLs: {e}")
            
            # Try to log system PATH
            logger.info(f"Current PATH: {os.environ.get('PATH', 'Not available')}")
            
            # Check if we're running from a PyInstaller bundle
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                logger.info(f"Running from PyInstaller bundle: {getattr(sys, '_MEIPASS', 'Unknown')}")
                
                # Check for expected DLL locations in PyInstaller bundle
                internal_lib_path = os.path.join(os.path.dirname(sys.executable), '_internal', 'llama_cpp', 'lib')
                if os.path.exists(internal_lib_path):
                    logger.info(f"Found internal lib directory: {internal_lib_path}")
                    if os.path.exists(os.path.join(internal_lib_path, 'llama.dll')):
                        logger.info(f"Found llama.dll in internal lib directory")
                    else:
                        logger.error(f"llama.dll NOT FOUND in internal lib directory")
                else:
                    logger.error(f"Internal lib directory not found: {internal_lib_path}")
            
            # Try to log which DLLs are loaded
            try:
                import psutil
                process = psutil.Process()
                logger.info("Attempting to list loaded DLLs in current process:")
                found_llama_dll = False
                for dll in process.memory_maps():
                    if '.dll' in dll.path.lower():
                        logger.info(f"  Loaded DLL: {dll.path}")
                        if 'llama.dll' in dll.path.lower():
                            found_llama_dll = True
                
                if not found_llama_dll:
                    logger.error("CRITICAL: llama.dll is not loaded in the current process!")
            except Exception as e:
                logger.error(f"Failed to list loaded DLLs: {e}")
        
        try:
            import llama_cpp
        except ImportError:
            logger.error("llama-cpp-python is not installed, cannot initialize backend.")
            return False

        # Check if the backend init function is already patched by gui_app.py
        if hasattr(llama_cpp, 'llama_backend_init'):
            logger.info("Using backend initialization function from llama_cpp")
            try:
                llama_cpp.llama_backend_init()  # Call with default False argument
                logger.info("Backend initialized successfully using llama_backend_init")
                return True
            except OSError as e:
                logger.error(f"OSError in llama_backend_init: {e}")
                if "access violation reading" in str(e):
                    logger.error("This is likely a DLL loading issue - wrong DLL version or architecture (32/64-bit mismatch)")
                    logger.error("Make sure you're using 64-bit Python and 64-bit llama-cpp-python")
                return False
            except Exception as e:
                logger.error(f"Error calling llama_backend_init: {e}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                return False
        else:
            # Try finding the function in llama_cpp module structure
            try:
                # Log llama_cpp module structure for debugging
                logger.info(f"llama_cpp module attributes: {dir(llama_cpp)}")
                
                # First try with direct _lib
                if hasattr(llama_cpp, '_lib') and hasattr(getattr(llama_cpp, '_lib'), 'llama_backend_init'):
                    lib = getattr(llama_cpp, '_lib')
                    logger.info("Found backend init in llama_cpp._lib")
                # Then try with nested module
                elif hasattr(llama_cpp, 'llama_cpp') and hasattr(llama_cpp.llama_cpp, '_lib') and hasattr(llama_cpp.llama_cpp._lib, 'llama_backend_init'):
                    lib = llama_cpp.llama_cpp._lib
                    logger.info("Found backend init in llama_cpp.llama_cpp._lib")
                else:
                    logger.warning("Could not find llama_backend_init in any known location")
                    return False
                    
                # Set correct argument types
                import ctypes
                lib.llama_backend_init.argtypes = [ctypes.c_bool]
                lib.llama_backend_init.restype = None
                
                # Call with default argument
                try:
                    lib.llama_backend_init(ctypes.c_bool(False))
                    logger.info("Backend initialized successfully using direct _lib call")
                    return True
                except OSError as e:
                    logger.error(f"OSError in direct llama_backend_init call: {e}")
                    if "access violation reading" in str(e):
                        logger.error("This is likely a DLL loading issue - wrong DLL version or architecture (32/64-bit mismatch)")
                        logger.error("Make sure you're using 64-bit Python and 64-bit llama-cpp-python")
                    return False
            except Exception as e:
                logger.error(f"Error initializing backend: {e}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                return False

    def load_model(self, model_path):
        logger.debug(f"load_model called with model_path: {model_path}")
        try:
            logger.info(f"Attempting to load model from {model_path}")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # More aggressive cleanup of previous model
            if hasattr(self, 'loaded_model') and self.loaded_model is not None:
                logger.info("Unloading previous model...")
                try:
                    # Force destruction of any GPU context
                    self.loaded_model.__del__()
                except Exception as e:
                    logger.warning(f"Error calling __del__ on model: {e}")
                
                # Set to None and delete reference
                del self.loaded_model
                self.loaded_model = None
                
                # Force garbage collection
                gc.collect()
                
                # Add a small delay to ensure GPU resources are released
                import time
                time.sleep(1)
                
                logger.debug("Previous model unloaded and garbage collected.")

            # Skip _safe_init_backend entirely - let Llama handle it
            from llama_cpp import Llama
            
            # ALWAYS start with CPU-only mode first to ensure it works
            n_ctx_to_use = min(self.n_ctx, 2048)  # Limit context to prevent OOM
            
            params = {
                "model_path": model_path,
                "n_ctx": n_ctx_to_use,
                "n_batch": 512, 
                "verbose": False,  # Reduced verbosity
                "n_gpu_layers": 0  # FORCE CPU ONLY for first load attempt
            }
            logger.info(f"Loading model with Llama params: {params}")
            
            # First check if the llama_cpp DLLs are accessible
            try:
                import llama_cpp
                logger.info(f"llama_cpp module found at: {llama_cpp.__file__}")
                dll_dir = os.path.join(os.path.dirname(llama_cpp.__file__), 'lib')
                logger.info(f"llama_cpp library directory should be at: {dll_dir}")
                if os.path.exists(dll_dir):
                    logger.info(f"Library directory exists with contents: {os.listdir(dll_dir)}")
                    # Add DLL directory to path for Windows
                    if platform.system() == "Windows":
                        # Try adding the DLL directory to the path
                        import ctypes
                        ctypes.cdll.LoadLibrary(os.path.join(dll_dir, "llama.dll"))
                        logger.info(f"Successfully loaded llama.dll from {dll_dir}")
                else:
                    logger.warning(f"Library directory does not exist at {dll_dir}")
                    # Try to find DLLs in alternative locations
                    exe_dir = os.path.dirname(sys.executable)
                    logger.info(f"Checking for DLLs in executable directory: {exe_dir}")
                    if os.path.exists(exe_dir):
                        logger.info(f"Executable directory contents: {os.listdir(exe_dir)}")
            except Exception as e:
                logger.error(f"Error while checking llama_cpp setup: {e}")
                logger.error(traceback.format_exc())
            
            # This is the critical part - let Llama handle its own initialization
            try:
                self.loaded_model = Llama(**params)
                self.model_path = model_path
                logger.info(f"Model '{os.path.basename(model_path)}' loaded successfully using Llama class.")
            except OSError as e:
                if "access violation reading" in str(e):
                    logger.error(f"Access violation error in llama_cpp - likely missing or incompatible DLLs: {e}")
                    # Try to recover by forcing static loading
                    try:
                        logger.info("Attempting fallback initialization approach...")
                        # Force Python to find DLLs in current directory
                        original_dir = os.getcwd()
                        os.chdir(os.path.dirname(sys.executable))
                        logger.info(f"Changed directory to {os.getcwd()} to find DLLs")
                        self.loaded_model = Llama(**params)
                        self.model_path = model_path
                        logger.info(f"Fallback approach successful! Model loaded.")
                        # Change back to original directory
                        os.chdir(original_dir)
                    except Exception as fallback_e:
                        logger.error(f"Fallback initialization also failed: {fallback_e}")
                        raise
                else:
                    raise
            
            # If the CPU-only load worked and user wanted GPU layers, try again with GPU
            if self.n_gpu_layers > 0:
                logger.info(f"CPU-only load successful, attempting with {self.n_gpu_layers} GPU layers")
                try:
                    # Save the CPU model just in case
                    cpu_model = self.loaded_model
                    
                    # Try with requested GPU layers
                    gpu_params = params.copy()
                    gpu_params["n_gpu_layers"] = self.n_gpu_layers
                    logger.info(f"Loading model with GPU params: {gpu_params}")
                    
                    self.loaded_model = Llama(**gpu_params)
                    logger.info("Successfully loaded model with GPU acceleration")
                    
                    # If GPU load succeeded, we can delete the CPU model
                    del cpu_model
                    gc.collect()
                except Exception as gpu_error:
                    logger.warning(f"GPU load failed, falling back to CPU model: {str(gpu_error)}")
                    # We already have a working CPU model, so just keep using that
        
            self._extract_model_metadata()  # FIXED: This should be outside the if-block
            return True
    
        except Exception as e:
            logger.exception(f"CRITICAL ERROR loading model '{os.path.basename(model_path)}'")
            self.loaded_model = None 
            self.model_path = None
            return False

    def _extract_model_metadata(self):
        logger.debug("Attempting to extract model metadata.")
        if not self.loaded_model:
            logger.warning("No model loaded, cannot extract metadata.")
            self.model_metadata = {}
            return
        
        try:
            metadata = {}
            # llama-cpp-python 0.2.20+ has model.metadata()
            if hasattr(self.loaded_model, 'metadata') and callable(self.loaded_model.metadata):
                logger.debug("Using model.metadata() method.")
                metadata_dict = self.loaded_model.metadata()
                if isinstance(metadata_dict, dict):
                    metadata.update(metadata_dict)
            
            # Fallback or additional info from attributes
            try:
                train_ctx_method = getattr(self.loaded_model, 'n_ctx_train', None)
                if callable(train_ctx_method):
                    train_ctx = train_ctx_method()
                    metadata['n_ctx_train'] = train_ctx
                    logger.debug(f"Extracted n_ctx_train: {train_ctx}")
            except Exception as e:
                logger.debug("Attribute n_ctx_train not accessible; skipping extraction.")

            if hasattr(self.loaded_model, 'n_embd'):
                 embd_size = self.loaded_model.n_embd() if callable(self.loaded_model.n_embd) else self.loaded_model.n_embd
                 metadata['n_embd'] = embd_size
                 logger.debug(f"Extracted n_embd: {embd_size}")
            
            if self.model_path: # Add filename from the stored path
                metadata['filename'] = os.path.basename(self.model_path)
            
            self.model_metadata = metadata
            logger.info(f"Extracted model metadata: {self.model_metadata if metadata else 'No specific metadata found.'}")
        except Exception as e:
            logger.exception("Error extracting model metadata")
            self.model_metadata = {"error": "Failed to extract metadata."}

    def health(self):
        # First check if the model object exists
        is_healthy = hasattr(self, 'loaded_model') and self.loaded_model is not None
        model_path = getattr(self, 'model_path', None)
        logger.debug(f"Health check: Model loaded = {is_healthy}, model_path = {model_path}")
        
        # Check model path existence and attributes
        if not is_healthy and model_path and os.path.exists(model_path):
            logger.warning(f"Model path exists but model not loaded. Attempting to reload from: {model_path}")
            
            # Try loading the model with a clean state
            try:
                # Clear any partial model state
                if hasattr(self, 'loaded_model') and self.loaded_model:
                    del self.loaded_model
                    self.loaded_model = None
                    gc.collect()
                
                # Try to reload the model
                is_healthy = self.load_model(model_path)
                logger.info(f"Reload attempt result: {is_healthy}")
                
                # Double-verify the model is now loaded
                is_healthy = hasattr(self, 'loaded_model') and self.loaded_model is not None
            except Exception as e:
                logger.error(f"Error during health check reload: {str(e)}")
                is_healthy = False
            
        return is_healthy

    def generate(self, prompt, max_tokens=2000, options=None):
        logger.debug(f"Generate called with prompt (type: {type(prompt)}), max_tokens: {max_tokens}")
        
        if not hasattr(self, 'loaded_model') or self.loaded_model is None:
            model_path = getattr(self, 'model_path', 'None')
            logger.error(f"Generation failed: No model loaded. Model path: {model_path}")
            return "Error: No model loaded. Please load a model first."
        
        options = options or {}        # Check for custom system prompt in options
        system_message_content = options.get("system_message")
        
        # Use default prompt if no custom prompt provided
        if not system_message_content:
            system_message_content = """You are an AI assistant named Ladbon AI, created by Ladbon Fragari. 
            You are a helpful assistant that provides accurate, direct responses.

IMPORTANT INSTRUCTIONS:
1. DO NOT invent or imagine user questions
2. DO NOT refer to previous messages that weren't actually sent
3. DO NOT include 'Ladbon AI:' or any prefix in your responses
4. DO NOT ask follow-up questions unless necessary
5. Format code with proper indentation using markdown code blocks
6. Be direct, accurate, and helpful

Remember: Only respond to what was actually asked, never make up user messages."""

        # Format the prompt according to its type
        if isinstance(prompt, str):
            simple_prompt = f"System: {system_message_content}\n\nUser: {prompt}\nAssistant:"
        elif isinstance(prompt, list): 
            simple_prompt = f"System: {system_message_content}\n\n"
            for msg in prompt:
                role = msg.get("role", "")
                content = msg.get("content", "")
                simple_prompt += f"{role.capitalize()}: {content}\n\n"
            simple_prompt += "Assistant:"
        else:
            logger.error(f"Invalid prompt type for generate: {type(prompt)}")
            return "Error: Invalid prompt format."
            
        generation_settings = {
            "max_tokens": max_tokens,
            "temperature": options.get("temperature", 0.7),
            "top_p": options.get("top_p", 0.9),
            "stop": options.get("stop", []),
        }
        
        try:
            logger.info(f"Generating with simplified prompt for small model")
            response_stream = self.loaded_model.create_completion(
                prompt=simple_prompt,
                max_tokens=generation_settings["max_tokens"],
                temperature=generation_settings["temperature"],
                top_p=generation_settings["top_p"],
                stop=generation_settings["stop"],
                stream=True
            )
            
            # Correctly accumulate content from all chunks in the stream
            content = ""
            for chunk in response_stream:
                try:
                    # First check if chunk is string (simple text chunk)
                    if isinstance(chunk, str):
                        content += chunk
                        continue
                        
                    # Handle dictionary format (most common)
                    if isinstance(chunk, dict):
                        if "choices" in chunk and chunk["choices"] and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            if isinstance(choice, dict):
                                text = choice.get("text", "")
                                content += text
                            elif hasattr(choice, "text"):
                                content += choice.text
                        else:
                            logger.debug(f"Received chunk with unusual format: {chunk}")
                    # Handle direct text format
                    elif "text" in chunk:
                        content += chunk["text"]
                    # Handle object with attributes format
                    elif hasattr(chunk, "choices"):
                        if hasattr(chunk.choices, "__len__") and len(chunk.choices) > 0:
                            choice = chunk.choices[0]
                            if hasattr(choice, "text"):
                                content += choice.text
                    # Handle direct text attribute
                    elif hasattr(chunk, "text"):
                        content += chunk.text
                    else:
                        logger.warning(f"Unexpected chunk format: {type(chunk)}")
                except Exception as e:
                    logger.exception(f"Error processing response chunk: {str(e)}")
                    # Continue with next chunk even if there's an error
            
            logger.debug(f"Raw response: {content[:200]}...")
            return content.strip()

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.exception(f"Error generating response: {str(e)}\n{error_trace}")
            return f"Error generating response: {str(e)}\n\nTry restarting the application or using a different model."

    def list_models(self):
        logger.debug("Listing models from local directories.")
        models = []
        # Use the utility function to get models directory
        app_models_dir = get_models_dir()
        # Legacy paths for backward compatibility
        standard_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        user_models_dir = os.path.join(Path.home(), ".ladbon_ai", "models")
        
        # Primary search location is now in AppData
        search_dirs = [app_models_dir]
        
        # Add legacy paths for backward compatibility
        if app_models_dir != standard_models_dir:  # Only add if they're different
            search_dirs.append(standard_models_dir)
        search_dirs.append(user_models_dir)
        
        # Add custom path from environment if specified
        if "LADBON_AI_MODELS_PATH" in os.environ:
            env_path = os.environ["LADBON_AI_MODELS_PATH"]
            search_dirs.append(env_path)
            logger.debug(f"Added LADBON_AI_MODELS_PATH to search: {env_path}")

        for models_dir_path in search_dirs:
            logger.debug(f"Searching for models in: {models_dir_path}")
            if os.path.exists(models_dir_path) and os.path.isdir(models_dir_path):
                for file in os.listdir(models_dir_path):
                    if file.lower().endswith(('.gguf', '.ggml')): # Common model extensions
                        model_id = self._get_model_id_from_filename(file)
                        models.append(model_id)
                        logger.debug(f"Found model file: {file}, identified as ID: {model_id}")
            else:
                logger.debug(f"Directory not found or not a directory: {models_dir_path}")
        
        unique_sorted_models = sorted(list(set(models)))
        logger.info(f"Discovered local models (IDs): {unique_sorted_models}")
        return unique_sorted_models

    def _get_model_id_from_filename(self, filename):
        name_part = filename.lower().split('.')[0]  # Remove extension
    
        # Handle special cases
        if "llama-2" in name_part:
            return "llama"
        elif "tinyllama" in name_part:
            return "tinyllama"
        elif "medgemma" in name_part:
            return "medgemma"
    
        # Default fallback
        model_id = name_part.split('-')[0]
        logger.debug(f"Extracted model ID '{model_id}' from filename '{filename}' (original name_part: '{name_part}')")
        return model_id


    def map_model_to_file(self, model_id, models_dir_to_search):
        logger.debug(f"Attempting to map model ID '{model_id}' to a GGUF/GGML file in directory '{models_dir_to_search}'")
        
        if not os.path.exists(models_dir_to_search):
            logger.warning(f"Directory '{models_dir_to_search}' does not exist for mapping model ID '{model_id}'.")
            return None
        
        # First check for exact match
        exact_path = os.path.join(models_dir_to_search, f"{model_id}.gguf")
        if os.path.exists(exact_path):
            logger.info(f"Found exact match for model ID '{model_id}': {exact_path}")
            return exact_path
        
        # Check for partial matches
        for file in os.listdir(models_dir_to_search):
            if file.lower().endswith(('.gguf', '.ggml')):
                file_lower = file.lower()
                model_id_lower = model_id.lower()
                
                # Log what we're checking
                logger.debug(f"Checking file '{file}'. Looking for model ID '{model_id}'.")
                
                # Try different match patterns
                if file_lower.startswith(model_id_lower):
                    logger.info(f"Found file starting with model ID '{model_id}': {file}")
                    return os.path.join(models_dir_to_search, file)
                elif model_id_lower in file_lower:
                    logger.info(f"Found file containing model ID '{model_id}': {file}")
                    return os.path.join(models_dir_to_search, file)
                elif model_id_lower == "llama" and "llama-2" in file_lower:
                    logger.info(f"Found Llama 2 file for model ID '{model_id}': {file}")
                    return os.path.join(models_dir_to_search, file)
                
                # Extract model ID from filename for comparison
                derived_id = self._get_model_id_from_filename(file)
                logger.debug(f"Checking file '{file}'. Derived ID: '{derived_id}'. Target ID: '{model_id}'.")
                if derived_id == model_id:
                    logger.info(f"Found matching file for model ID '{model_id}': {os.path.join(models_dir_to_search, file)}")
                    return os.path.join(models_dir_to_search, file)
        
        logger.warning(f"No matching GGUF file found for model ID '{model_id}' in '{models_dir_to_search}' after all checks.")
        return None
        
    def get_model_info(self, model_file_path):
        logger.debug(f"Getting info for model file: {model_file_path}")
        # ... (keep existing logic, add logging) ...
        if not os.path.exists(model_file_path):
            logger.error(f"File not found for get_model_info: {model_file_path}")
            return {"error": "File not found"}
        # Extract file info
        info = {}
        info["filename"] = os.path.basename(model_file_path)
        info["size_mb"] = round(os.path.getsize(model_file_path) / (1024 * 1024), 2)
        logger.info(f"Model info for '{os.path.basename(model_file_path)}': Size {info['size_mb']} MB")
        return info
    
    def is_model_compatible(self, model_path):
        logger.debug(f"Checking compatibility for model: {model_path}")
        # ... (keep existing logic, add logging) ...
        # This is a placeholder. True compatibility requires deeper inspection.
        model_name_lower = os.path.basename(model_path).lower()
        if "gemma" in model_name_lower:
            # Example: Gemma models might need newer llama-cpp-python
            if self.llamacpp_version.startswith("0.1.") or self.llamacpp_version.startswith("0.2.0") or self.llamacpp_version.startswith("0.2.1"):
                logger.warning(f"Model '{model_path}' (Gemma) might be incompatible with llama-cpp-python version {self.llamacpp_version}.")
                return False, f"Gemma model likely incompatible with version {self.llamacpp_version}"
        logger.debug(f"Basic compatibility check passed for '{model_path}'.")
        return True, "Compatibility check passed (basic)."

    def switch_model(self, model_id):
        """Switch to a specified model by ID"""
        logger.info(f"Switching to model: {model_id}")
        try:
            # Primary location: AppData models directory
            app_models_dir = get_models_dir()
            
            # Legacy locations for backward compatibility
            standard_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            user_models_dir = os.path.join(Path.home(), ".ladbon_ai", "models")
            
            # First try AppData location
            target_model_path = self.map_model_to_file(model_id, app_models_dir)
            
            # Then try legacy locations if needed
            if not target_model_path:
                # Search in alternate directories
                search_dirs = [standard_models_dir, user_models_dir]
                
                # Add env var path if specified
                if "LADBON_AI_MODELS_PATH" in os.environ:
                    search_dirs.append(os.environ["LADBON_AI_MODELS_PATH"])
                
                for s_dir in search_dirs:
                    path_attempt = self.map_model_to_file(model_id, s_dir)
                    if path_attempt:
                        target_model_path = path_attempt
                        break

            if not target_model_path:
                logger.error(f"Could not find a model file for model ID: {model_id}")
                return False

            logger.info(f"Mapped model ID '{model_id}' to file: {target_model_path}")
            
            # If we already have this exact model loaded, don't reload it
            if self.loaded_model and self.model_path == target_model_path:
                logger.info(f"Model '{model_id}' ({target_model_path}) is already loaded. No switch needed.")
                return True

            # Unloading is handled by load_model if a model is already loaded.
            logger.info(f"Proceeding to load new model: {target_model_path}")
            success = self.load_model(target_model_path)
            
            # Double-check that the model was actually loaded
            model_actually_loaded = hasattr(self, 'loaded_model') and self.loaded_model is not None
            if success and not model_actually_loaded:
                logger.error(f"Model loading reported success but model object is None! Path: {target_model_path}")
                success = False
                
            if success:
                logger.info(f"Successfully switched to and loaded model: {model_id} ({target_model_path})")
            else:
                logger.error(f"Failed to load model {model_id} ({target_model_path}) during switch operation.")
                
            return success
            
        except Exception as e:
            logger.exception(f"Error during model switch to '{model_id}'")
            return False

    def check_model_loading_diagnostic(self, model_path_to_check):
        logger.info(f"Running model loading diagnostic for: {model_path_to_check}")
        # ... (keep existing logic, add logging within the try/excepts) ...
        results = {
            "python_version": platform.python_version(),
            "os_platform": f"{platform.system()} {platform.release()}",
            "llamacpp_version": self.llamacpp_version, # Ensure this is set
            "model_path_checked": model_path_to_check,
            "model_exists": None,
            "model_size_mb": None,
            "available_memory_gb": None,
            "backend_init_simulated": "Patch in gui_app.py handles main init. Llama() constructor calls it.",
            "minimal_load_test_status": "Not run",
            "minimal_load_test_error": None
        }
        logger.debug(f"Initial diagnostic results structure: {results}")
        
        try:
            import psutil # Moved import here
            results["available_memory_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
            logger.debug(f"Available memory: {results['available_memory_gb']} GB")
        except ImportError:
            logger.warning("psutil not installed, cannot report available memory.")
            results["available_memory_gb"] = "psutil not installed"
        
        if not model_path_to_check or not isinstance(model_path_to_check, str):
            results["minimal_load_test_error"] = "Invalid model_path_to_check provided."
            logger.warning("Diagnostic check: Invalid model_path_to_check.")
            return results

        results["model_exists"] = os.path.exists(model_path_to_check)
        if results["model_exists"]:
            results["model_size_mb"] = round(os.path.getsize(model_path_to_check) / (1024 * 1024), 2)
            logger.info(f"Diagnostic check: Model '{model_path_to_check}' exists, size {results['model_size_mb']} MB.")
            
            # Minimal load test
            temp_model = None
            try:
                from llama_cpp import Llama # Ensure import
                logger.info(f"Attempting minimal load test for {model_path_to_check} with n_gpu_layers=0.")
                temp_model = Llama(model_path=model_path_to_check, n_gpu_layers=0, verbose=False, n_ctx=512)
                results["minimal_load_test_status"] = "Success (CPU)"
                logger.info("Minimal load test (CPU) successful.")
            except Exception as e:
                logger.exception("Minimal load test (CPU) FAILED.")
                results["minimal_load_test_status"] = "Failed (CPU)"
                results["minimal_load_test_error"] = str(e)
            finally:
                if temp_model is not None:
                    del temp_model
                    gc.collect()
                    logger.debug("Cleaned up temporary model from diagnostic test.")
        else:
            results["minimal_load_test_error"] = "Model file does not exist."
            logger.warning(f"Diagnostic check: Model file '{model_path_to_check}' does not exist.")
            
        logger.info(f"Final diagnostic results: {results}")
        return results

# Example of how you might use the client (optional, for testing)
if __name__ == '__main__':
    main_logger = setup_logger('llamacpp_client_standalone_test')
    main_logger.info("Running LlamaCppClient standalone test...")
    client = LlamaCppClient()
    client.n_gpu_layers = 0 
    
    main_logger.info(f"Llama.cpp Version: {client.llamacpp_version}")
    main_logger.info(f"Using API v3: {client.is_v3_api}")

    available_models = client.list_models()
    print(f"Available models found: {available_models}")

    if available_models:
        model_to_test = available_models[0] # Test the first model found
        print(f"\nAttempting to switch to and test model: {model_to_test}")
        
        # Find the actual path for the diagnostic
        standard_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        user_models_dir = os.path.join(Path.home(), ".ladbon_ai", "models")
        model_file_for_diag = client.map_model_to_file(model_to_test, standard_models_dir) or \
                              client.map_model_to_file(model_to_test, user_models_dir)

        if model_file_for_diag:
            print(f"\nRunning diagnostic for: {model_file_for_diag}")
            diag_results = client.check_model_loading_diagnostic(model_file_for_diag)
            print("Diagnostic Results:")
            for key, value in diag_results.items():
                print(f"  {key}: {value}")
        else:
            print(f"Could not find file for model ID {model_to_test} to run diagnostic.")


        if client.switch_model(model_to_test):
            print(f"\nSuccessfully switched to {model_to_test}.")
            print(f"Model metadata: {client.model_metadata}")
            
            prompt_text = "Explain quantum physics in simple terms."
            print(f"\nGenerating response for prompt: '{prompt_text}'")
            response = client.generate(prompt_text, max_tokens=150)
            print("\nGenerated Response:")
            print(response)
        else:
            print(f"\nFailed to switch to model {model_to_test}.")
    else:
        print("\nNo models found to test. Please ensure models are in the 'models' directory or ~/.ladbon_ai/models.")

    print("\nStandalone test finished.")