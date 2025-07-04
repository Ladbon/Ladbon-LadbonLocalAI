import logging
import llama_cpp
import sys

# Configure logging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

def initialize_llama_backend():
    """
    Initializes the llama.cpp backend.
    This must be called before any other llama_cpp functions.
    """
    try:
        logger.info("Attempting to initialize llama_cpp backend...")
        # The 'complain_if_already_initialized' parameter is not valid.
        # We simply call the function. If it's already initialized, it should handle it gracefully.
        llama_cpp.llama_backend_init()
        logger.info("llama_cpp backend initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize llama_cpp backend: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # This allows running the script directly for testing if needed.
    initialize_llama_backend()
