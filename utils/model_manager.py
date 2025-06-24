import os
from huggingface_hub import hf_hub_download
from utils.logger import setup_logger # Import logger

logger = setup_logger(__name__) # Setup logger for this module

class ModelManager:
    def __init__(self, models_dir=None):
        """Initialize the model manager with correct repositories"""
        import os
        
        # Set models directory
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize client references as None
        self.llamacpp_client = None
        self.ollama_client = None
        
        # Define available models with CORRECTED sizes
        self.available_models = {
            "llama2-7b-chat": { # Changed from "llama2-7b"
                "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                "filename": "llama-2-7b-chat.Q4_K_M.gguf",
                "description": "Llama 2 7B (Chat)",
                "size_mb": 3900
            },
            "phi3-mini": {
                "repo_id": "TheBloke/phi-3-mini-4k-instruct-GGUF",
                "filename": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
                "description": "Phi-3 Mini (Chat)",
                "size_mb": 1800
            },
            "mistral-7b": {
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "description": "Mistral 7B Instruct v0.2",
                "size_mb": 4500
            },
            "tinyllama": {
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "description": "TinyLlama 1.1B (Fast Chat)",
                "size_mb": 640  # Corrected from 700MB to 640MB based on actual file size
            }
        }
        
        logger.info(f"ModelManager initialized. Models directory: {self.models_dir}")
        
    def get_installed_models(self):
        """Get a list of installed models"""
        import os
        
        installed = []
        for model_id, info in self.available_models.items():
            model_path = os.path.join(self.models_dir, info["filename"])
            if os.path.exists(model_path):
                installed.append({"id": model_id, "path": model_path})
        
        logger.debug(f"Found installed models: {installed}")
        return installed

    def get_available_models(self):
        """Get dictionary of available models"""
        logger.debug("Returning list of available (predefined) models.")
        return self.available_models
    
    def download_model(self, model_id, progress_callback=None):
        """Download a model using hf_hub_download directly."""
        logger.info(f"Request to download model_id: {model_id}")
        if model_id not in self.available_models:
            logger.error(f"Model ID {model_id} not found in predefined available_models.")
            return False, f"Model {model_id} not found in available models"
            
        model_info = self.available_models[model_id]
        os.makedirs(self.models_dir, exist_ok=True)
        logger.debug(f"Ensured models directory exists: {self.models_dir}")
        
        output_path = os.path.join(self.models_dir, model_info["filename"])
        
        if os.path.exists(output_path):
            logger.info(f"Model {model_id} (file: {model_info['filename']}) already downloaded at {output_path}")
            # Ensure progress callback reflects completion if model already exists
            if progress_callback:
                progress_callback(100) # Simulate 100%
            return True, f"Model {model_id} already downloaded"
        
        try:
            logger.info(f"Starting download for model: {model_id} from repo: {model_info['repo_id']} to file: {model_info['filename']}")
            
            hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                local_dir=self.models_dir,
                local_dir_use_symlinks=False, # Good for Windows
                resume_download=True,
                # etag_timeout=10 # Default, consider increasing if timeouts occur
            )
            
            if os.path.exists(output_path):
                logger.info(f"Successfully downloaded {model_id} to {output_path}")
                if progress_callback: 
                    progress_callback(100) # Simulate 100% completion
                return True, f"Successfully downloaded {model_id}"
            else:
                # This case should be rare if hf_hub_download doesn't raise an error
                logger.error(f"Download reported success but file not found at {output_path} for model {model_id}")
                return False, f"Download failed for {model_id}, file not found post-download."

        except Exception as e:
            logger.exception(f"Error downloading model {model_id}") # Use logger.exception
            return False, f"Error downloading {model_id}: {str(e)}"
    
    def set_llamacpp_client(self, client):
        """Set the reference to the LlamaCppClient instance"""
        self.llamacpp_client = client
        logger.info("LlamaCppClient reference set in ModelManager")
        
    def set_ollama_client(self, client):
        """Set the reference to the OllamaClient instance"""
        self.ollama_client = client
        logger.info("OllamaClient reference set in ModelManager")
        
    def get_active_client(self):
        """Return the active client that has a model loaded"""
        if hasattr(self, 'llamacpp_client') and self.llamacpp_client and self.llamacpp_client.health():
            return self.llamacpp_client
        elif hasattr(self, 'ollama_client') and self.ollama_client and self.ollama_client.health():
            return self.ollama_client
        else:
            logger.error("No active clients available or no models loaded")
            return None