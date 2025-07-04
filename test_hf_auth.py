"""
Test script for Hugging Face authentication
"""
import os
import sys
import logging

# Add the parent directory to the path to allow importing from utils
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf_auth_test")

try:
    # Import our Hugging Face auth utilities
    from utils.hf_auth import setup_huggingface, is_hf_logged_in, load_hf_token
    
    # Set up Hugging Face with proper cache location and load token if available
    setup_huggingface()
    
    # Check if we're logged in
    if is_hf_logged_in():
        token = load_hf_token()
        logger.info("Successfully authenticated with Hugging Face!")
        logger.info("Token is available: %s", "Yes" if token else "No")
        
        # Try to access whoami to verify token
        from huggingface_hub import whoami
        try:
            user_info = whoami(token=token)
            logger.info("User info: %s", user_info)
        except Exception as e:
            logger.error("Failed to get user info: %s", e)
    else:
        logger.warning("Not authenticated with Hugging Face. Please add your token in the Settings tab.")
        
    # Check cache directory
    from utils.data_paths import get_app_data_dir
    app_data_dir = get_app_data_dir()
    hf_cache_dir = os.path.join(app_data_dir, "hf_cache")
    logger.info("Hugging Face cache directory: %s", hf_cache_dir)
    logger.info("Cache directory exists: %s", os.path.exists(hf_cache_dir))
    
    # List any existing models in the models directory
    from utils.data_paths import get_models_dir
    models_dir = get_models_dir()
    logger.info("Models directory: %s", models_dir)
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.gguf')]
        logger.info("Found %d GGUF models in the models directory", len(models))
        for model in models:
            logger.info("  - %s", model)
    else:
        logger.warning("Models directory does not exist")
    
    logger.info("Test completed successfully")
except Exception as e:
    logger.exception("An error occurred during testing")
    sys.exit(1)
