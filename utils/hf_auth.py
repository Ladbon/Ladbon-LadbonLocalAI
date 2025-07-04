"""
Hugging Face authentication utility module.
Handles token management for Hugging Face downloads.
"""
import os
import json
import logging
from pathlib import Path
from huggingface_hub import login, whoami
from huggingface_hub.utils._hf_folder import HfFolder
from utils.data_paths import get_app_data_dir

logger = logging.getLogger(__name__)

def get_hf_token_path():
    """Get the path where HF token should be stored"""
    app_data_dir = get_app_data_dir()
    return os.path.join(app_data_dir, "hf_token.json")

def save_hf_token(token, token_name="Ladbon_AI_Token"):
    """Save the Hugging Face token to a file in the app data directory"""
    token_path = get_hf_token_path()
    try:
        # First, let's validate the token by attempting to log in
        logger.info("Attempting to validate Hugging Face token...")
        
        # Store token and set in environment
        with open(token_path, 'w') as f:
            json.dump({
                "token": token,
                "name": token_name
            }, f)
        
        # Set it in the environment for the current session
        os.environ["HF_TOKEN"] = token
        
        # Try to actually login with the token
        login(token=token, add_to_git_credential=False)
        
        # If login succeeded, get user info
        try:
            user_info = whoami(token=token)
            username = user_info.get('name', 'Unknown')
            logger.info(f"Hugging Face authentication successful - Logged in as: {username}")
            return True, f"Token saved successfully - Logged in as: {username}"
        except Exception as e:
            logger.error(f"Token saved but couldn't get user info: {str(e)}")
            return True, "Token saved successfully, but couldn't verify user info"
    
    except Exception as e:
        logger.error(f"Failed to save Hugging Face token: {str(e)}")
        # If the token file was created but login failed, try to remove it
        if os.path.exists(token_path):
            try:
                os.remove(token_path)
                logger.info("Removed invalid token file")
            except:
                pass
        return False, f"Failed to save token: {str(e)}"

def load_hf_token():
    """Load the Hugging Face token from the app data directory"""
    token_path = get_hf_token_path()
    if not os.path.exists(token_path):
        return None
    
    try:
        with open(token_path, 'r') as f:
            data = json.load(f)
            token = data.get("token")
            if token:
                # Set in environment for this session
                os.environ["HF_TOKEN"] = token
            return token
    except Exception as e:
        logger.error(f"Failed to load Hugging Face token: {str(e)}")
        return None

def is_hf_logged_in():
    """Check if we have a valid HF token and are logged in"""
    token = load_hf_token()
    if not token:
        return False
    
    try:
        # Try to access user info to verify token
        user_info = whoami(token=token)
        return user_info is not None
    except Exception:
        return False

def clear_hf_token():
    """Clear the saved Hugging Face token"""
    token_path = get_hf_token_path()
    if os.path.exists(token_path):
        try:
            os.remove(token_path)
            if "HF_TOKEN" in os.environ:
                del os.environ["HF_TOKEN"]
            return True, "Token removed successfully"
        except Exception as e:
            logger.error(f"Failed to remove Hugging Face token: {str(e)}")
            return False, f"Failed to remove token: {str(e)}"
    return True, "No token found"

def configure_huggingface_cache():
    """Configure Hugging Face cache directory in appdata location"""
    app_data_dir = get_app_data_dir()
    hf_cache_dir = os.path.join(app_data_dir, "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    # Set the cache directory for this session
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
    
    return hf_cache_dir

def setup_huggingface():
    """Set up Hugging Face with proper cache location and load token if available"""
    # Configure cache directory
    hf_cache_dir = configure_huggingface_cache()
    logger.info(f"Configured Hugging Face cache directory: {hf_cache_dir}")
    
    # Try to load and use token if available
    token = load_hf_token()
    if token:
        try:
            # Don't add to git credential to avoid issues with non-dev environments
            logger.info("Token found, attempting to log in to Hugging Face...")
            login(token=token, add_to_git_credential=False, write_permission=True)
            
            # Get user info to verify successful login
            user_info = whoami(token=token)
            username = user_info.get('name', 'Unknown')
            user_id = user_info.get('id', 'Unknown')
            
            logger.info(f"Successfully logged in to Hugging Face as: {username} (ID: {user_id})")
            
            # Log organization info if available
            if 'orgs' in user_info and user_info['orgs']:
                orgs = [org.get('name', 'Unknown') for org in user_info['orgs']]
                logger.info(f"User belongs to organizations: {', '.join(orgs)}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {str(e)}")
            import traceback
            logger.debug(f"Hugging Face login error details: {traceback.format_exc()}")
    else:
        logger.info("No Hugging Face token found")
    
    return False
