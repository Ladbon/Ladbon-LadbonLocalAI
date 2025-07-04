import os
import requests
import base64
from utils.logger import setup_logger
import traceback
from typing import Optional
import shutil
import sys # Importing sys here

# Create a logger for this module
logger = setup_logger('ollama_client')

class OllamaClient:
    def __init__(self, address: Optional[str] = None, connect_on_init=True):
        logger.info(f"Initializing OllamaClient. Address: {address if address else 'default http://localhost:11434'}. Connect on init: {connect_on_init}")
        
        # Set environment variables for GPU optimization
        os.environ["OLLAMA_CUDA_MALLOC"] = "1"
        os.environ["OLLAMA_NUM_GPU_LAYERS"] = "99" # Push everything possible to GPU
        
        if address is None:
            self.address = "http://localhost:11434"
        else:
            self.address = address
        
        # Ensure Ollama is in PATH
        self._ensure_ollama_in_path()
        
        # Only check connection if explicitly requested
        if connect_on_init:
            self.health(show_error=False)

    def _ensure_ollama_in_path(self):
        """Ensure ollama.exe is in PATH, try common install locations if not."""
        ollama_name = "ollama.exe" if os.name == "nt" else "ollama"
        if shutil.which(ollama_name):
            logger.info(f"Ollama binary found in PATH: {shutil.which(ollama_name)}")
            return
        # Try common install locations
        possible_dirs = [
            os.path.expandvars(r"%ProgramFiles%\\Ollama"),
            os.path.expandvars(r"%ProgramFiles(x86)%\\Ollama"),
            os.path.expandvars(r"%LocalAppData%\\Programs\\Ollama"),
            os.path.expanduser(r"~\\AppData\\Local\\Programs\\Ollama"),
            os.path.expanduser(r"~/.ollama"),
            os.path.dirname(sys.executable),
        ]
        for d in possible_dirs:
            candidate = os.path.join(d, ollama_name)
            if os.path.isfile(candidate):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                logger.info(f"Added Ollama binary directory to PATH: {d}")
                return
        logger.warning("Ollama binary not found in PATH or common locations. Please install Ollama and ensure it is in your PATH.")

    def health(self, show_error=True) -> bool:
        """Check if Ollama is running, with optional error suppression"""
        logger.debug(f"Checking Ollama health at {self.address}/api/tags")
        try:
            # Added short timeout to prevent long hangs when Ollama is not running
            response = requests.get(f"{self.address}/api/tags", timeout=2.0)
            if response.status_code == 200:
                # Successfully connected to Ollama
                logger.info("Ollama health check successful.")
                
                # Verify we can get a list of models as an additional health check
                model_list = self.list_models()
                if model_list:
                    logger.info(f"Ollama has {len(model_list)} models available.")
                    return True
                else:
                    if show_error:
                        logger.warning("Ollama responded but no models are available.")
                    # Still consider healthy even if no models, since Ollama itself is running
                    return True
            else:
                if show_error:
                    logger.error(f"❌ Ollama returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError as e:
            if show_error:
                logger.error(f"Could not connect to Ollama at {self.address}: {e}")
            return False
        except Exception as e:
            if show_error:
                logger.error(f"❌ Error connecting to Ollama: {str(e)}")
            return False
            
    def generate(self, model: str, prompt: str, max_tokens: int = 8000, timeout: Optional[int] = None, options: Optional[dict] = None) -> str:
        logger.info(f"Ollama generate request. Model: {model}, Max Tokens: {max_tokens}")
        logger.debug(f"Prompt for Ollama: {prompt[:100]}...") # Log snippet
        """Generate chat completions with personalized creator attribution"""
        
        # Check for custom system prompt in options
        custom_system_prompt = None
        if options and "system_message" in options:
            custom_system_prompt = options.get("system_message")
            logger.info("Using custom system prompt from options")
            
        # Add system prompt with thinking/answer separation instructions
        system_message = {
            "role": "system", 
            "content": custom_system_prompt if custom_system_prompt else """You are an AI assistant named Ladbon AI, created by Ladbon Fragari. 
            You are a helpful assistant that provides accurate, direct responses.
            
        When responding:
        - Answer directly without prefacing with "Ladbon AI:" or any other prefix
        - DO NOT invent or hallucinate previous user messages or questions
        - Respond ONLY to the exact message provided by the user
        - Don't ask follow-up questions unless absolutely necessary
        - When greeting users, be warm but brief without asking how you can help
        - Format responses clearly with appropriate markdown
        
        For code examples:
        - Use proper indentation (4 spaces for Python, 2 spaces for most other languages)
        - Include proper opening and closing brackets/braces
        - Include necessary namespace/import statements at the top
        - Place each statement on its own line
        - Follow common style conventions for the language
        - Triple backticks must be on their own lines
        
        Remember: NEVER invent user questions that weren't actually asked."""
        }
        
        # Check for history in options
        messages = [system_message]
        if options and "history" in options:
            history = options.get("history", [])
            logger.info(f"Using conversation history of {len(history)} messages")
            messages.extend(history)
        else:
            # Default user message if no history
            user_message = {
                "role": "user", 
                "content": prompt
            }
            messages.append(user_message)
        
        # Use provided options or defaults
        default_options = {
            "max_tokens": max_tokens,
            "num_gpu": 1,
            "num_thread": 6,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 20,
            "seed": 42,
            "repeat_penalty": 1.1
        }
        
        # Merge provided options with defaults
        payload_options = default_options.copy()
        if options:
            payload_options.update(options)
          # Build payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": payload_options
        }
        
        try:
            resp = requests.post(f"{self.address}/v1/chat/completions", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            
            # Assuming 'content' is the raw response from the AI
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.debug(f"Ollama raw response content: {content[:200]}...") # Log snippet
            think_start = content.find("<think>")
            think_end = content.find("</think>")

            final_answer = content
            thinking_process = ""

            if think_start != -1 and think_end != -1 and think_end > think_start:
                thinking_process = content[think_start + len("<think>") : think_end].strip()
                final_answer = content[think_end + len("</think>") :].strip()
                
                # Now you have 'thinking_process' and 'final_answer'
                # You can choose to log thinking_process but only display final_answer
                logger.debug(f"AI Thinking: {thinking_process}")
            
            logger.info("Generation successful.")
            # Return only the final answer to be displayed
            return final_answer.strip() 
        except Exception as e:
            logger.exception(f"Error during Ollama generation for model {model}")
            return ""

    def list_models(self):
        logger.debug(f"Listing Ollama models from {self.address}")
        """Get list of models installed on Ollama"""
        models_result = []
        try:
            # Try using the 'ollama list' command directly
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the output format from the command line
                for line in result.stdout.splitlines()[1:]:  # Skip header line
                    if line.strip():
                        parts = line.strip().split()
                        if parts:
                            models_result.append(parts[0])  # First column is the model name
                logger.info(f"Found {len(models_result)} models using CLI command.")
                return models_result
        except Exception as e:
            logger.warning(f"Failed to get models via CLI: {str(e)}")
    
        # If CLI fails, try the API endpoints as before
        try:
            response = requests.get(f"{self.address}/v1/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Extract model names
                models_result = [model['name'] for model in data.get('models', [])]
                logger.info(f"Found {len(models_result)} models using v1/models endpoint.")
                return models_result
        except Exception as e:
            logger.warning(f"Failed to get models from API: {str(e)}")
        
        # If everything fails, return a default model
        logger.warning("No models found via CLI or API - returning default")
        return models_result # return the actual list
    def generate_with_image(self, model: str, prompt: str, image_path: str, max_tokens: int = 2048, timeout: Optional[int] = None, options: Optional[dict] = None) -> str:
        logger.info(f"Ollama image generation request. Model: {model}, Image: {image_path}")
        logger.debug(f"Prompt for Ollama image gen: {prompt[:100]}...")
        """Generate text with image input for multimodal models"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                error_msg = f"Image file not found: {image_path}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Log file info
            file_size = os.path.getsize(image_path) / 1024
            logger.info(f"Image file size: {file_size:.2f} KB")
              # Get model family for proper formatting
            is_mistral = "mistral" in model.lower()
            is_llava = "llava" in model.lower() or "bakllava" in model.lower()
            
            # Check for custom system prompt in options
            custom_system_prompt = None
            if options and "system_message" in options:
                custom_system_prompt = options.get("system_message")
                logger.info("Using custom system prompt from options for image generation")
            
            # System message to identify the creator
            system_message = {
                "role": "system", 
                "content": custom_system_prompt if custom_system_prompt else """You are an AI assistant named Ladbon AI, created by Ladbon Fragari. 
            You are a helpful assistant that provides accurate, direct responses.
            
        When responding:
        - Answer directly without prefacing with "Ladbon AI:" or any other prefix
        - DO NOT invent or hallucinate previous user messages or questions
        - Respond ONLY to the exact message provided by the user
        - Don't ask follow-up questions unless absolutely necessary
        - When greeting users, be warm but brief without asking how you can help
        - Format responses clearly with appropriate markdown
        
        For code examples:
        - Use proper indentation (4 spaces for Python, 2 spaces for most other languages)
        - Include proper opening and closing brackets/braces
        - Include necessary namespace/import statements at the top
        - Place each statement on its own line
        - Follow common style conventions for the language
        - Triple backticks must be on their own lines
        
        Remember: NEVER invent user questions that weren't actually asked."""
            }
            
            # Format based on exact model type
            if is_mistral:
                # Mistral format
                logger.info("Using Mistral format")
                messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data]
                    }
                ]
            elif is_llava:
                # Try the standard OpenAI multi-modal format for LLaVa
                logger.info("Using LLaVa OpenAI-style format")
                messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            else:
                # Default format for other models
                logger.info("Using default multimodal format")
                messages = [
                    system_message,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image_data}
                        ]
                    }
                ]
            
            # API request
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_ctx": 4096,
                    "num_gpu": 1,
                    "temperature": 0.2,
                    "max_tokens": max_tokens
                }
            }
            
            logger.debug("Sending request to Ollama API")
            resp = requests.post(f"{self.address}/v1/chat/completions", json=payload, timeout=timeout)
            resp.raise_for_status()
            logger.debug("Response received successfully")
            
            data = resp.json()
            response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.debug(f"Ollama image gen raw response: {response[:200]}...")
            logger.info("Image generation completed successfully")
            return response
            
        except requests.exceptions.RequestException as e:
            error_trace = traceback.format_exc()
            logger.error(f"API request error: {str(e)}")
            logger.error(error_trace)
            return f"Error connecting to Ollama API: {str(e)}"
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error in image generation: {str(e)}")
            logger.error(error_trace)
            return f"Error processing image: {str(e)}"
    def ensure_ollama_installed(self):
        """Check if Ollama is installed on the system, with improved error reporting."""
        import subprocess
        self._ensure_ollama_in_path()
        try:
            result = subprocess.run(["ollama", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return True
            else:
                logger.error(f"Ollama found but failed to run: {result.stderr.decode(errors='ignore')}")
                print("❌ Ollama is installed but failed to run. Check your installation.")
                return False
        except FileNotFoundError:
            logger.error("Ollama is not installed or not found in PATH.")
            print("❌ Ollama is not installed. Please install from https://ollama.ai/download and ensure it is in your PATH.")
            return False