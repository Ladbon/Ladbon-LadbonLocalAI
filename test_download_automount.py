"""
Test script to verify automatic model refresh and mounting after download.
This script tests the following workflow:
1. Initialize the app
2. Download a model
3. Verify that the model list is updated
4. Verify that the new model is auto-selected and mounted
"""

import os
import sys
import time
from PyQt5.QtWidgets import QApplication
from api.app import LocalAIApp
from utils.logger import setup_logger
from utils.data_paths import get_models_dir

logger = setup_logger('test_download_automount')

# Force this test to use integrated backend
os.environ['LADBON_AI_BACKEND'] = 'integrated'

def run_test():
    logger.info("Starting test: download_automount")
    
    app = QApplication(sys.argv)
    window = LocalAIApp()
    
    # Get initial model list
    initial_models = []
    if hasattr(window, 'llamacpp_client'):
        initial_models = window.llamacpp_client.list_models()
    
    logger.info(f"Initial models: {initial_models}")
    
    # Print current state
    logger.info(f"Current model directory: {get_models_dir()}")
    
    # Identify a model that doesn't exist locally
    test_model_id = "tinyllama"  # Small model for faster testing
    
    # If the test model already exists, log that
    if test_model_id in initial_models:
        logger.info(f"Test model {test_model_id} already exists. Will verify overwrite behavior.")
    
    # Force download the test model
    # For testing purposes, we'll manually call the methods that would be triggered by button clicks
    from utils.huggingface_manager import HuggingFaceManager
    # Ensure HuggingFaceManager is initialized
    if not hasattr(window, 'huggingface_manager'):
        window.huggingface_manager = HuggingFaceManager(models_dir=get_models_dir())
    
    # First, select the model in the available models list
    # Find model ID in items - more robust checking of item data
    model_found = False
    for i in range(window.available_models_list.count()):
        item = window.available_models_list.item(i)
        if item:
            item_data = item.data(0)
            item_text = item.text().lower()
            logger.info(f"Examining item {i}: text='{item.text()}', data={item_data}")
            
            # Try to match both the data and text containing the model ID
            if (item_data and item_data == test_model_id) or test_model_id.lower() in item_text:
                window.available_models_list.setCurrentItem(item)
                model_found = True
                logger.info(f"Selected model {test_model_id} in available models list (item {i})")
                break
    
    if not model_found:
        logger.error(f"Could not find model {test_model_id} in available models list!")
        # Try to select any model as a fallback
        if window.available_models_list.count() > 0:
            window.available_models_list.setCurrentItem(window.available_models_list.item(0))
            logger.info(f"Selected first available model as fallback")
    
    # Make sure we have a model selected
    selected_items = window.available_models_list.selectedItems()
    if not selected_items:
        logger.error("No model selected! Cannot proceed with download test.")
        return
        
    selected_model = selected_items[0].text()
    logger.info(f"Starting download for selected model: {selected_model}...")
    
    # Trigger the download directly
    if hasattr(window, 'download_button') and window.download_button.isEnabled():
        window.download_button.click()
        logger.info("Clicked download button")
    else:
        logger.error("Download button not available or disabled")
        
    # We need to make sure the UI is properly updated to show download progress
    # For that, the event loop should be active during download
    def check_download_status():
        # This is a simulation of real usage - the download happens in a separate thread
        # We just check periodically to see if the UI shows it's completed
        
        max_checks = 50  # Prevent infinite loop
        checks = 0
        
        while checks < max_checks:
            checks += 1
            
            # Process events to update UI
            QApplication.processEvents()
            
            # Check if model status indicates completion
            status_text = window.model_status_label.text() if hasattr(window, 'model_status_label') else ""
            
            if "Download complete" in status_text:
                logger.info("Download completed!")
                break
                
            time.sleep(1)  # Wait a second between checks
        
        # After download (or timeout), check the final state
        final_models = []
        if hasattr(window, 'llamacpp_client'):
            final_models = window.llamacpp_client.list_models()
        
        logger.info(f"Final models: {final_models}")
        
        # Get the current model ID from the UI
        current_model = None
        if hasattr(window, 'model_combo') and window.model_combo.count() > 0:
            current_index = window.model_combo.currentIndex()
            if current_index >= 0:
                current_model = window.model_combo.itemData(current_index)
                
        logger.info(f"Current model after download: {current_model}")
        
        # Define success criteria:
        # 1. The model should be in the list of final models
        # 2. The model should be auto-selected in the UI
        # 3. The model should be loaded in LlamaCppClient
        
        # Check if test model is in the list
        if test_model_id in final_models or any(test_model_id.lower() in m.lower() for m in final_models):
            logger.info(f"SUCCESS: Model {test_model_id} is in the final list.")
        else:
            logger.error(f"FAILURE: Model {test_model_id} is NOT in final list.")
        
        # Check if the model was auto-selected
        if current_model and (current_model == test_model_id or test_model_id.lower() in current_model.lower()):
            logger.info(f"SUCCESS: Model {test_model_id} was auto-selected in the UI.")
        else:
            logger.error(f"FAILURE: Test model {test_model_id} was NOT auto-selected. Current: {current_model}")
            
        # Check if the model was auto-mounted in LlamaCppClient
        if hasattr(window, 'llamacpp_client'):
            model_path = window.llamacpp_client.model_path
            logger.info(f"LlamaCpp model path: {model_path}")
            
            if model_path and test_model_id.lower() in model_path.lower():
                logger.info(f"SUCCESS: Model {test_model_id} was properly mounted in LlamaCpp.")
            else:
                logger.error(f"FAILURE: Model {test_model_id} was NOT properly mounted in LlamaCpp.")
        
        # Check if current model was auto-selected
        current_model = window.current_model if hasattr(window, 'current_model') else None
        logger.info(f"Current model after download: {current_model}")
        
        if current_model == test_model_id:
            logger.info(f"SUCCESS: Test model {test_model_id} was auto-selected.")
        else:
            logger.error(f"FAILURE: Test model {test_model_id} was NOT auto-selected. Current: {current_model}")
        
        # Check if model is properly mounted (loaded in LlamaCpp)
        llamacpp_model_path = window.llamacpp_client.model_path if hasattr(window.llamacpp_client, 'model_path') else None
        logger.info(f"LlamaCpp model path: {llamacpp_model_path}")
        
        if llamacpp_model_path and test_model_id.lower() in llamacpp_model_path.lower():
            logger.info(f"SUCCESS: Model {test_model_id} was properly mounted in LlamaCpp.")
        else:
            logger.error(f"FAILURE: Model {test_model_id} was NOT properly mounted in LlamaCpp.")
            
        # Clean up
        app.quit()
    
    # Actually trigger the download
    window.force_download_model()
    
    # Set up a timer to check download status after a short delay
    from PyQt5.QtCore import QTimer
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(check_download_status)
    timer.start(2000)  # Start checking after 2 seconds
    
    # Start event loop (this will block until app.quit() is called)
    app.exec_()
    
    logger.info("Test completed!")

if __name__ == "__main__":
    run_test()
