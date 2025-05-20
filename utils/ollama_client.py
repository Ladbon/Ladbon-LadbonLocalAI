import os
import requests
import base64
from utils.logger import setup_logger
import traceback

# Create a logger for this module
logger = setup_logger('ollama_client')

class OllamaClient:
    def __init__(self, address: str = None):
        # Set environment variables for GPU optimization
        os.environ["OLLAMA_CUDA_MALLOC"] = "1"
        os.environ["OLLAMA_NUM_GPU_LAYERS"] = "99" # Push everything possible to GPU
        
        # Use env var or default
        self.address = address or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.address}/v1/models")
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, model: str, prompt: str, max_tokens: int = 8000, timeout: int = None, options: dict = None) -> str:
        """Generate chat completions with personalized creator attribution"""
        # Add system prompt identifying you as the creator - keep it short for speed
        system_message = {
            "role": "system", 
            "content": "You are LAIdbon, an AI assistant created by Ladbon Fragari."
        }
        
        user_message = {
            "role": "user", 
            "content": prompt
        }
        
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
            "messages": [system_message, user_message],
            "stream": False,
            "options": payload_options
        }
        
        resp = requests.post(f"{self.address}/v1/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def generate_with_image(self, model: str, prompt: str, image_path: str, max_tokens: int = 2048, timeout: int = None) -> str:
        """Generate text with image input for multimodal models"""
        try:
            logger.info(f"Image generation request with model: {model}")
            logger.info(f"Image path: {image_path}")
            logger.debug(f"Prompt: {prompt}")
            
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
            
            # System message to identify the creator
            system_message = {
                "role": "system", 
                "content": "You are an AI assistant created by Ladbon Fragari. Your name is Ladbon AI."
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
        """Check if Ollama is installed on the system"""
        import subprocess
        import sys
        
        try:
            subprocess.run(["ollama", "version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            print("❌ Ollama is not installed. Please install from https://ollama.ai/download")
            return False