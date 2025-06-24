import os
import sys
import logging
import datetime

def setup_logger(name='localai'):
    """Set up and return a logger that writes to both a file and the console"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    log_file = os.path.join(logs_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')  # Use UTF-8 encoding
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a custom formatter that handles encoding errors
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            # Format the message
            formatted_message = super().format(record)
            # Replace characters that might cause encoding issues in console
            if isinstance(formatted_message, str):
                try:
                    # Test if it can be encoded in cp1252
                    formatted_message.encode('cp1252')
                except UnicodeEncodeError:
                    # If not, replace problematic characters
                    formatted_message = formatted_message.encode('cp1252', errors='replace').decode('cp1252')
            return formatted_message
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = SafeFormatter('%(levelname)s: %(message)s')
    
    # Add formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    logger.info(f"Logging started to {log_file}")
    return logger

# Create a default logger instance
logger = setup_logger()