import sys
import os
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QTabWidget, QFileDialog, QLabel, QMessageBox,
                             QComboBox, QProgressBar, QStatusBar, QCheckBox,
                             QListWidget, QListWidgetItem, QSplitter, QFormLayout, QSpinBox,
                             QGroupBox, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QTextCursor, QIcon
import time 
import gc  # Add garbage collection module
from utils.ollama_client import OllamaClient
from utils.data_paths import get_settings_path, get_docs_dir, get_img_dir, get_logs_dir, get_models_dir, first_run_migration, get_app_data_dir
from utils.hf_auth import setup_huggingface, is_hf_logged_in, save_hf_token, clear_hf_token, load_hf_token
from cli.doc_handler import process_document
from cli.img_handler import process_image, query_image
from cli.web_search import search
from utils.logger import setup_logger
import traceback

logger = setup_logger('LocalAIApp_GUI') # Main GUI logger

from utils.huggingface_manager import HuggingFaceManager
from utils.llamacpp_client import LlamaCppClient

# Check if llama-cpp-python is working properly
def check_llamacpp_installation():
    """Check llama-cpp-python installation and diagnose common issues"""
    try:
        from llama_cpp import Llama
        print("llama-cpp-python is installed")
        
        # Try to initialize the backend (this fails if there's an issue)
        import ctypes
        import llama_cpp as llama_cpp
        try:
            llama_cpp.llama_backend_init()
            print("llama_cpp backend initialized successfully")
            return True
        except OSError as e:
            print(f"Error initializing llama_cpp backend: {e}")
            print("\nPossible solutions:")
            print("1. Run reinstall_llamacpp.py to reinstall llama-cpp-python")
            print("2. Make sure your GPU drivers are up to date")
            print("3. Try using CPU-only mode by setting n_gpu_layers=0")
            return False
    except ImportError:
        print("llama-cpp-python is not installed")
        return False

MODEL_CAPABILITIES = {
    "llama4:7b": {
        "description": "Llama 4 7B (Chat, Docs, Vision, Web)",
        "doc": True, 
        "img": True,
        "ocr": True,
        "web": True,
        "rag": True
    },
    # Mistral models
    "mistral:v1": {
        "description": "Mistral v1 (Chat, Docs)",
        "doc": True,
        "img": False,
        "ocr": False,
        "web": True,
        "rag": False
    },
    "mistral-small3.1:24b": {
        "description": "Mistral Small 3.1 (Chat, Docs, RAG)",
        "doc": True,
        "img": False,
        "ocr": False,
        "web": True,
        "rag": True
    },
    # Qwen models
    "qwen3:1.8b": {
        "description": "Qwen 3 1.8B (Fast Chat, Web)",
        "doc": False,
        "img": False,
        "ocr": False,
        "web": True,
        "rag": False
    },
    "qwen3:8b": {
        "description": "Qwen 3 8B (Chat, Docs, OCR, Web)",
        "doc": True,
        "img": False,
        "ocr": True,
        "web": True,
        "rag": True
    },
    # LLaVA models
    "llava:7b": {
        "description": "LLaVA 7B (Vision)",
        "doc": False,
        "img": True,
        "ocr": True,
        "web": False,
        "rag": False
    },
    # Gemma models
    "gemma:2b": {
        "description": "Gemma 2B (Fast Chat)",
        "doc": False,
        "img": False,
        "ocr": False,
        "web": False,
        "rag": False
    },
    "gemma:7b": {
        "description": "Gemma 7B (Chat, Docs)",
        "doc": True,
        "img": False,
        "ocr": False,
        "web": True, 
        "rag": False
    },
    # Phi models
    "phi3:3.8b": {
        "description": "Phi-3 3.8B (Chat, Vision)",
        "doc": False,
        "img": True,
        "ocr": True,
        "web": True,
        "rag": False
    },
    "phi3:14b": {
        "description": "Phi-3 14B (Chat, Vision, Docs)",
        "doc": True,
        "img": True,
        "ocr": True,
        "web": True,
        "rag": True
    },
    # Deepseek models
    "deepseek:7b": {
        "description": "DeepSeek 7B (Chat, Docs)",
        "doc": True,
        "img": False,
        "ocr": False,
        "web": True,
        "rag": False
    }
}

class GenerationThread(QThread):
    """Thread for running the text generation"""
    finished = pyqtSignal(str)
    import os  # Add os import for path operations
    
    def __init__(self, client, model, prompt, max_tokens, timeout, options=None):
        """Initialize the generation thread with client, model, prompt, and options"""
        super().__init__()
        self.client = client
        self.model_id_str = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.options = options or {}
        
    def run(self):
        # First check if we have a valid client
        if not self.client:
            self.finished.emit("Error: No AI model client available. Please ensure a model is selected and loaded.")
            return
            
        # Check if client has a loaded model, specifically for llamacpp_client
        if hasattr(self.client, 'health') and not self.client.health():
            model_path = getattr(self.client, 'model_path', 'unknown')
            self.finished.emit(f"Error: No model loaded. The model at {model_path} needs to be reloaded. Try switching models or restarting the application.")
            return
            
        try:
            # Format the conversation for the model
            # If options contains history, use it to format conversations properly
            messages = []
            
            # Add system message if available
            if self.options and "system_message" in self.options:
                messages.append({
                    "role": "system", 
                    "content": self.options["system_message"]
                })
            
            # Add history if available
            if "history" in self.options and self.options["history"]:
                messages.extend(self.options["history"])
            
            # Add current prompt as user message
            messages.append({"role": "user", "content": self.prompt})
            
            # For Ollama client, pass the model_id
            if hasattr(self.client, 'address') and 'ollama' in str(type(self.client)).lower():
                # Extra check for model argument with Ollama
                if not self.model_id_str:
                    self.finished.emit("Error: No model selected for Ollama. Please select a model first.")
                    return
                    
                response = self.client.generate(
                    model=self.model_id_str,
                    prompt=self.prompt,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                    options=self.options
                )
            else:
                # LlamaCPP client
                generation_options = self.options.copy()
                # Set history in the format LlamaCPP expects
                if len(messages) > 1:  # If we have more than just the user message
                    generation_options["history"] = messages
                
                # Double check model is loaded
                if not hasattr(self.client, 'loaded_model') or self.client.loaded_model is None:
                    model_path = getattr(self.client, 'model_path', None)
                    if model_path and os.path.exists(model_path):
                        # Try to reload the model
                        success = self.client.load_model(model_path)
                        if not success:
                            self.finished.emit(f"Error: Failed to reload model at {model_path}. Try restarting the application.")
                            return
                    else:
                        self.finished.emit("Error: No model is loaded. Please select a model first.")
                        return
                        
                response = self.client.generate(
                    prompt=self.prompt,
                    max_tokens=self.max_tokens,
                    options=generation_options
                )
            self.finished.emit(response)
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.finished.emit(error_msg)

class ImageAnalysisThread(QThread):
    """Thread for analyzing images"""
    finished = pyqtSignal(str)
    def __init__(self, client, model, prompt, image_path, max_tokens=2048, options=None, timeout=None):
        super().__init__()
        self.client = client
        self.model = model
        self.prompt = prompt
        self.image_path = image_path
        self.max_tokens = max_tokens
        self.options = options or {}
        self.timeout = timeout
        
    def run(self):
        try:
            response = self.client.generate_with_image(
                model=self.model,
                prompt=self.prompt,
                image_path=self.image_path,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                options=self.options
            )
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class LocalAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize status bar early
        self.setStatusBar(QStatusBar())
        
        # Initialize logger
        self.logger = logger # Use the module-level logger or re-assign

        # Enable verbose logging
        import logging
        self.logger.setLevel(logging.DEBUG)

        self.logger.info("LocalAIApp GUI application starting")
        
        # Check for first run and migrate data if needed
        first_run_migration()
        self.logger.info(f"Using app data directory: {get_app_data_dir()}")
        
        self.ui_initialized = False 
        
        # Initialize with integrated backend first
        self.backend = "integrated"
        from utils.model_manager import ModelManager
        self.model_manager = ModelManager()
        
        # Initialize variables before UI
        self.history = []
        self.selected_docs = []
        self.selected_imgs = []
        self.web_search_enabled = False
        self.generation_thread = None
        
        # Default values, will be overridden by load_settings
        self.n_ctx_setting = 4096
        self.n_gpu_layers_setting = 0  # Default to CPU for safer loading
        self.force_cpu_only = True  # Default to CPU-only mode for stability

        # Default config - DEFINE THESE BEFORE init_ui()
        self.current_model = "llama"  # Use a simple model name that matches your files
        self.max_tokens = 8192
        self.timeout = None
        
        # Create Llamacpp client with CPU-only mode for initial setup
        self.llamacpp_client = LlamaCppClient(n_ctx=4096, n_gpu_layers=0)
        self.model_manager.set_llamacpp_client(self.llamacpp_client)
        self.client = self.llamacpp_client
        
        # Now initialize the UI after variables are set
        self.init_ui()
        self.ui_initialized = True 
        
        # Load initial logs
        self.load_logs()
        
        # Setup HuggingFace authentication and cache
        setup_huggingface()
        self.logger.info("HuggingFace authentication setup completed")
        
        # Only AFTER UI is ready, create Ollama client
        from utils.ollama_client import OllamaClient
        self.ollama_client = OllamaClient(connect_on_init=False)  
        
        # Load settings
        self.load_settings()
        
        # Force CPU mode for safety - once everything else is working, you can try GPU again
        self.force_cpu_only = True
        
        # Recreate llamacpp client with settings
        self.llamacpp_client = LlamaCppClient(
            n_ctx=self.n_ctx_setting,
            n_gpu_layers=0 if self.force_cpu_only else self.n_gpu_layers_setting
        )
        self.model_manager.set_llamacpp_client(self.llamacpp_client)
        
        # Update UI based on loaded model capabilities
        self.update_ui_for_model_capabilities()
        
        # Set up log auto-refresh timer
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.load_logs)
        self.log_timer.start(1000)  # Refresh every second

        # Try to automatically load a model if any are found - ONLY ONCE
        self.try_autoload_model()

    def try_autoload_model(self):
        """Try to automatically load a model safely"""
        try:
            # Ensure this uses the correct client
            current_client_for_list = self.llamacpp_client if self.backend == "integrated" else self.ollama_client
            installed_models = current_client_for_list.list_models()

            if installed_models:
                # Attempt to load the model that was saved in settings, or the first available one
                model_to_load = self.current_model  # From loaded settings
                if model_to_load not in installed_models:
                    model_to_load = installed_models[0]  # Fallback to first

                self.logger.info(f"Attempting to automatically load model on startup: {model_to_load}")
                
                # Clear memory before loading
                gc.collect()
                
                if self.backend == "integrated":
                    # Force CPU loading for safety
                    self.llamacpp_client.n_gpu_layers = 0
                    
                    if self.llamacpp_client.switch_model(model_to_load):
                        self.logger.info(f"Successfully auto-loaded model: {model_to_load}")
                        self.current_model = model_to_load  # Ensure current_model is updated
                        
                        # Update combo box selection
                        idx = self.model_combo.findData(self.current_model)
                        if idx != -1:
                            self.model_combo.setCurrentIndex(idx)
                        else:  # Try by text if findData fails
                            for i in range(self.model_combo.count()):
                                if self.current_model in self.model_combo.itemText(i):
                                    self.model_combo.setCurrentIndex(i)
                                    break
                    else:
                        self.logger.error(f"Failed to auto-load model: {model_to_load}")
                elif self.backend == "ollama":
                    self.current_model = model_to_load
                    self.logger.info(f"Set current model to (Ollama): {model_to_load}")
                    idx = self.model_combo.findData(self.current_model)
                    if idx != -1:
                        self.model_combo.setCurrentIndex(idx)
                          # Update UI based on model capabilities
                self.update_ui_for_model_capabilities()
        except Exception as e:
            self.logger.error(f"Error during automatic model loading on startup: {str(e)}")
    
    def show_status_message(self, message: str, timeout: int = 0):
        """Safely show a message in the status bar with null check"""
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(message, timeout)
    
    def init_ui(self):
        """Initialize the user interface with document and image lists"""
        self.setWindowTitle("Ladbon AI Desktop - Created by Ladbon Fragari")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create a splitter for the main area
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Chat area
        chat_widget = self._create_chat_area()
        
        # Right side: Model selection and document/image lists
        right_widget = self._create_right_sidebar()
        
        # Add the widgets to the splitter
        self.main_splitter.addWidget(chat_widget)
        self.main_splitter.addWidget(right_widget)
        
        # Set the splitter sizes (70% chat, 30% sidebar)
        self.main_splitter.setSizes([700, 300])
        
        # Add splitter to main layout (top)
        main_layout.addWidget(self.main_splitter)
        
        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Load documents and images
        self.load_documents()
        self.load_images()

    def _create_chat_area(self):
        """Create the chat area widget"""
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        # Chat history display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.returnPressed.connect(self.send_message)
        
        # Add web search toggle button
        self.web_button = QPushButton("🔍 Web")
        self.web_button.setCheckable(True)
        self.web_button.clicked.connect(self.toggle_web_search)
        self.web_button.setToolTip("Enable web search for this message")

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.web_button)
        input_layout.addWidget(self.send_button)
        chat_layout.addLayout(input_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.hide()
        chat_layout.addWidget(self.progress)

        # Create and add tabs (now inside chat area, below chat)
        self._create_tabs()
        chat_layout.addWidget(self.tabs)
        
        return chat_widget

    def _create_right_sidebar(self):
        """Create the right sidebar widget"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Model selection
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Select Model:"))
        
        self.model_combo = QComboBox()
        self.update_model_list()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        
        right_layout.addLayout(model_layout)
        
        # Document list section
        doc_layout = QVBoxLayout()
        doc_layout.addWidget(QLabel("Documents:"))
        
        self.doc_list = QListWidget()
        self.doc_list.setSelectionMode(QListWidget.NoSelection)
        doc_layout.addWidget(self.doc_list)
        right_layout.addLayout(doc_layout)
        
        # Image list section
        img_layout = QVBoxLayout()
        img_layout.addWidget(QLabel("Images:"))
        
        self.img_list = QListWidget()
        self.img_list.setSelectionMode(QListWidget.NoSelection)
        img_layout.addWidget(self.img_list)
        right_layout.addLayout(img_layout)
        
        # Add logs section to the right sidebar
        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_group)
        
        self.logs_display = QTextEdit()
        self.logs_display.setReadOnly(True)
        self.logs_display.setLineWrapMode(QTextEdit.WidgetWidth)  # Enable word wrapping
        self.logs_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Disable horizontal scrolling
        logs_layout.addWidget(self.logs_display)
        
        # Add a refresh button for logs
        refresh_logs_btn = QPushButton("Refresh Logs")
        refresh_logs_btn.clicked.connect(self.load_logs)
        logs_layout.addWidget(refresh_logs_btn)
        
        right_layout.addWidget(logs_group)
        
        # Clear selections button
        clear_button = QPushButton("Clear Selections")
        clear_button.clicked.connect(self.clear_selections)
        right_layout.addWidget(clear_button)
          # Clear history button
        clear_history_button = QPushButton("Clear Chat History")
        clear_history_button.clicked.connect(self.clear_history)
        right_layout.addWidget(clear_history_button)        
        return right_widget

    def _create_tabs(self):
        """Create the tabs for settings and AI models"""
        self.tabs = QTabWidget()
        
        self.main_layout = QVBoxLayout()

        # Create and add the settings tab
        self.settings_tab = QWidget()
        self.init_settings_tab() # Initialize the comprehensive settings tab
        self.tabs.addTab(self.settings_tab, "Settings")  # Add the settings tab to the tab widget
        
        # Initialize model management tab
        self.model_management_tab = self.init_model_management_tab()
        self.tabs.addTab(self.model_management_tab, "AI Models")
        
    def init_settings_tab(self):
        """Initialize the settings tab for Llama.cpp parameters"""
        layout = QVBoxLayout(self.settings_tab)
        
        # Add explanatory text
        info_text = """<html><body>
        <h3>Llama.cpp Settings:</h3>
        <p>Adjust the parameters for Llama.cpp models. Recommended to use defaults unless you know the impact.</p>
        </body></html>"""
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Hugging Face Authentication Section
        from utils.hf_auth import is_hf_logged_in, save_hf_token, clear_hf_token
        
        hf_group = QGroupBox("Hugging Face Authentication")
        hf_layout = QVBoxLayout()
        
        hf_info_text = """<html><body>
        <p>Authenticate with Hugging Face to download models. Get your token from 
        <a href="https://huggingface.co/settings/tokens">https://huggingface.co/settings/tokens</a></p>
        </body></html>"""
        
        hf_info_label = QLabel(hf_info_text)
        hf_info_label.setOpenExternalLinks(True)
        hf_info_label.setWordWrap(True)
        hf_layout.addWidget(hf_info_label)
        
        hf_input_layout = QHBoxLayout()
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setEchoMode(QLineEdit.Password)  # Hide the token like a password
        self.hf_token_input.setPlaceholderText("Enter your Hugging Face token here")
        
        hf_save_btn = QPushButton("Save Token")
        hf_clear_btn = QPushButton("Clear Token")
        
        # Check current login status
        hf_status_label = QLabel()
        if is_hf_logged_in():
            hf_status_label.setText("Status: Authenticated with Hugging Face")
            hf_status_label.setStyleSheet("color: green")
        else:
            hf_status_label.setText("Status: Not authenticated")
            hf_status_label.setStyleSheet("color: red")
        
        hf_input_layout.addWidget(self.hf_token_input)
        hf_input_layout.addWidget(hf_save_btn)
        hf_input_layout.addWidget(hf_clear_btn)
        
        hf_layout.addLayout(hf_input_layout)
        hf_layout.addWidget(hf_status_label)
        
        # Connect buttons to functions
        def on_save_token():
            token = self.hf_token_input.text().strip()
            if not token:
                QMessageBox.warning(self, "Token Required", "Please enter a valid Hugging Face token.")
                return
            
            # Show a proper modal progress dialog during authentication
            progress_dialog = QProgressDialog("Authenticating with Hugging Face...", None, 0, 0, self)
            progress_dialog.setWindowTitle("Authenticating")
            progress_dialog.setModal(True)  # This is the correct way to make it modal
            progress_dialog.show()
            QApplication.processEvents()
            
            self.logger.info("Attempting to save Hugging Face token and authenticate...")
            
            try:
                success, message = save_hf_token(token)
                progress_dialog.close()  # Close the progress dialog
                self.show_status_message("")  # Clear the status message
                
                if success:
                    self.logger.info(f"Hugging Face authentication successful: {message}")
                    QMessageBox.information(self, "Success", f"Hugging Face authentication successful!\n{message}")
                    hf_status_label.setText(f"Status: {message}")
                    hf_status_label.setStyleSheet("color: green")
                    self.hf_token_input.clear()  # Clear the input for security
                else:
                    self.logger.error(f"Failed to authenticate with Hugging Face: {message}")
                    QMessageBox.critical(self, "Error", f"Failed to save token: {message}")
            except Exception as e:
                progress_dialog.close()  # Close the progress dialog
                self.show_status_message("")  # Clear the status message
                self.logger.exception(f"Exception during Hugging Face authentication: {str(e)}")
                QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        
        def on_clear_token():
            self.logger.info("Clearing Hugging Face token...")
            success, message = clear_hf_token()
            if success:
                self.logger.info("Hugging Face token removed successfully")
                QMessageBox.information(self, "Success", "Hugging Face token removed.")
                hf_status_label.setText("Status: Not authenticated")
                hf_status_label.setStyleSheet("color: red")
            else:
                self.logger.error(f"Failed to remove Hugging Face token: {message}")
                QMessageBox.critical(self, "Error", f"Failed to remove token: {message}")
        
        hf_save_btn.clicked.connect(on_save_token)
        hf_clear_btn.clicked.connect(on_clear_token)
        
        hf_group.setLayout(hf_layout)
        layout.addWidget(hf_group)
        
        # General Settings Section
        general_group = QGroupBox("General Settings")
        general_layout = QFormLayout()
        
        # System prompt settings
        self.system_prompt_label = QLabel("System Prompt:")
        self.system_prompt_input = QTextEdit()
        self.system_prompt_input.setPlaceholderText("Custom system prompt. Leave empty to use default.")
        self.system_prompt_input.setMinimumHeight(150)
        
        # Load existing custom system prompt if exists
        if hasattr(self, 'custom_system_prompt_text'):
            self.system_prompt_input.setPlainText(self.custom_system_prompt_text)
        
        general_layout.addRow(self.system_prompt_label, self.system_prompt_input)
        
        # Timeout setting
        self.timeout_input = QSpinBox()
        self.timeout_input.setRange(0, 6000)  # 0-6000 seconds
        timeout_value = 0
        if hasattr(self, 'timeout') and self.timeout is not None:
            timeout_value = self.timeout
        self.timeout_input.setValue(timeout_value)
        self.timeout_input.setSpecialValueText("None (No timeout)")
        self.timeout_input.setToolTip("Response timeout in seconds. 0 means no timeout.")
        general_layout.addRow("Response Timeout (seconds):", self.timeout_input)
          # Max tokens setting
        self.settings_tokens_input = QLineEdit(str(self.max_tokens))
        # Sync with the main tokens input if it exists
        if hasattr(self, 'tokens_input') and self.tokens_input:
            self.settings_tokens_input.setText(self.tokens_input.text())
        self.settings_tokens_input.setToolTip("Maximum number of tokens to generate in response")
        general_layout.addRow("Max Tokens:", self.settings_tokens_input)
        
        general_group.setLayout(general_layout)
        layout.addWidget(general_group)
        
        # Save general settings button
        save_settings_btn = QPushButton("Save General Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_settings_btn)
        
        # Llama.cpp Settings Section
        llamacpp_group = QGroupBox("Llama.cpp Settings")
        settings_layout = QFormLayout()
        
        self.n_ctx_input = QSpinBox()
        self.n_ctx_input.setRange(512, 32768) # Example range
        self.n_ctx_input.setValue(self.n_ctx_setting)
        self.n_ctx_input.setToolTip("Context window size for Llama.cpp models.")
        settings_layout.addRow("Context Size (n_ctx):", self.n_ctx_input)

        self.n_gpu_layers_input = QSpinBox()
        self.n_gpu_layers_input.setRange(-1, 200) # -1 for all layers, 0 for CPU
        self.n_gpu_layers_input.setValue(self.n_gpu_layers_setting)
        self.n_gpu_layers_input.setToolTip("Number of layers to offload to GPU (-1 for all, 0 for CPU only).")
        settings_layout.addRow("GPU Layers (n_gpu_layers):", self.n_gpu_layers_input)

        self.force_cpu_checkbox = QCheckBox("Force CPU Only (Overrides GPU Layers)")
        self.force_cpu_checkbox.setChecked(self.force_cpu_only)
        self.force_cpu_checkbox.setToolTip("If checked, n_gpu_layers will be set to 0, regardless of the value above.")
        settings_layout.addRow(self.force_cpu_checkbox)

        save_settings_button = QPushButton("Save Llama.cpp Settings")
        save_settings_button.clicked.connect(self.save_llamacpp_settings)
        llamacpp_group.setLayout(settings_layout)
        layout.addWidget(llamacpp_group)
        layout.addWidget(save_settings_button)
          # Set the layout for the settings tab
        self.settings_tab.setLayout(layout)
        
        # Re-select the current tab to refresh it
        current_index = self.tabs.currentIndex()
        self.tabs.setCurrentIndex(-1)
        self.tabs.setCurrentIndex(current_index)

    # markdown_lists_to_html method
    def markdown_lists_to_html(self, text):
        """Convert markdown lists to HTML with better styling"""
        import re
        
        if text.startswith("Error:"):
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = text.split('\n')
        html_lines = []
        in_list_ul = False 
        in_list_ol = False
        list_indent = 0
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check for unordered list items
            ul_match = re.match(r'^(\s*)([\*\-\+])\s+(.+)$', line)
            if ul_match:
                indent = len(ul_match.group(1))
                content = ul_match.group(3)
                
                if not in_list_ul or indent != list_indent:
                    if in_list_ol:
                        html_lines.append("</ol>")
                        in_list_ol = False
                    if in_list_ul and indent > list_indent:
                        # Nested list
                        html_lines.append("<ul style='margin-left: 20px;'>")
                    elif in_list_ul and indent < list_indent:
                        # End of nested list
                        html_lines.append("</ul>")
                    elif not in_list_ul:
                        html_lines.append("<ul style='margin: 10px 0; padding-left: 25px;'>")
                    
                    in_list_ul = True
                    list_indent = indent
                    
                html_lines.append(f"<li style='margin-bottom: 6px;'>{content}</li>")
                continue
                    
            # Check for ordered list items
            ol_match = re.match(r'^(\s*)(\d+)\.?\s+(.+)$', line)
            if ol_match:
                indent = len(ol_match.group(1))
                content = ol_match.group(3)
                
                if not in_list_ol or indent != list_indent:
                    if in_list_ul:
                        html_lines.append("</ul>")
                        in_list_ul = False
                    if in_list_ol and indent > list_indent:
                        # Nested list
                        html_lines.append("<ol style='margin-left: 20px;'>")
                    elif in_list_ol and indent < list_indent:
                        # End of nested list
                        html_lines.append("</ol>")
                    elif not in_list_ol:
                        html_lines.append("<ol style='margin: 10px 0; padding-left: 25px;'>")
                    
                    in_list_ol = True
                    list_indent = indent
                    
                html_lines.append(f"<li style='margin-bottom: 6px;'>{content}</li>")
                continue
                
            # Not a list item
            if in_list_ul:
                html_lines.append("</ul>")
                in_list_ul = False
            if in_list_ol:
                html_lines.append("</ol>")
                in_list_ol = False
            list_indent = 0
            
            html_lines.append(line)
        
        # Close any open lists
        if in_list_ul:
            html_lines.append("</ul>")
        if in_list_ol:
            html_lines.append("</ol>")
            
        return '\n'.join(html_lines)
    
    def format_ai_response(self, text):
        """Format AI response with enhanced styling and markdown support"""
        import re
        
        # 1. Handle code blocks with syntax highlighting
        code_pattern = r'```([\w]*)\n(.*?)\n```'
        text = re.sub(code_pattern, self._format_code_block, text, flags=re.DOTALL)
        
        # 2. Handle inline code
        inline_code_pattern = r'`(.*?)`'
        text = re.sub(inline_code_pattern, r'<code style="background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; font-family: monospace;">\1</code>', text)
        
        # 3. Handle headers
        text = re.sub(r'^# (.*?)$', r'<h1 style="color: #2c3e50; margin-top: 10px; margin-bottom: 10px;">\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*?)$', r'<h2 style="color: #2c3e50; margin-top: 10px; margin-bottom: 10px;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.*?)$', r'<h3 style="color: #2c3e50; margin-top: 10px; margin-bottom: 10px;">\1</h3>', text, flags=re.MULTILINE)
        
        # 4. Handle lists (both bulleted and numbered)
        text = self.markdown_lists_to_html(text)
        
        # 5. Handle bold and italic
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        # 6. Handle paragraphs
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        for p in paragraphs:
            if not p.startswith('<h') and not p.startswith('<ul') and not p.startswith('<ol') and not p.startswith('<pre'):
                p = f'<p style="margin-top: 8px; margin-bottom: 8px;">{p}</p>'
            formatted_paragraphs.append(p)
        text = '\n'.join(formatted_paragraphs)
        
        return text
    
    def add_to_chat(self, sender, message):
        """Add a message to the chat display with improved padding and alignment"""
        self.logger.debug(f"Adding to chat. Sender: {sender}, Message length: {len(message)}")
        
        # Common message box styling - increased overall padding
        box_style = "padding: 18px 20px; margin: 16px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); text-align: left;"
        
        if sender == "User":
            # User message styling - blue bubble with consistent spacing
            user_style = box_style + "background-color: #e3f2fd; margin-right: 40px; border-left: 4px solid #1976d2;"
            
            # Escape HTML in user messages for security
            escaped_message = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            self.chat_display.append(f"""
                <div style="{user_style}">
                    <div style="font-weight: bold; margin-bottom: 10px; color: #1976d2; text-align: left;">You:</div>
                    <div style="text-align: left; line-height: 1.5;">{escaped_message}</div>
                </div>
            """)
        else:
            # AI message styling - light gray bubble with consistent spacing
            ai_style = box_style + "background-color: #f5f5f5; margin-right: 40px; border-left: 4px solid #424242;"
            
            self.chat_display.append(f"""
                <div style="{ai_style}">
                    <div style="font-weight: bold; margin-bottom: 10px; color: #424242; text-align: left;">AI:</div>
                    <div style="text-align: left; line-height: 1.5;">{message}</div>
                </div>
            """)
        
        self.chat_display.ensureCursorVisible()

    def clear_selections(self):
        """Clear all document and image selections"""
        # Uncheck all documents
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            if item is not None:
                item.setCheckState(Qt.CheckState.Unchecked)
                
        # Uncheck all images
        for i in range(self.img_list.count()):
            item = self.img_list.item(i)
            if item is not None:
                item.setCheckState(Qt.CheckState.Unchecked)
                
        self.selected_docs = []
        self.selected_imgs = []
        
        self.show_status_message("Selections cleared")
        
    def clear_history(self):
        """Clear chat history"""
        self.history = []
        self.chat_display.clear()
        self.show_status_message("Chat history cleared")
    
    def load_settings(self):
        """Load user settings from a JSON file"""
        import json
        self.logger.debug("Loading settings...")
        try:
            settings_path = get_settings_path()  # Use our utility function
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    settings = json.load(f)
                    
                    # General settings
                    self.current_model = settings.get("model", "qwen3:8b")
                    self.max_tokens = settings.get("max_tokens", 8192)
                    
                    # Load custom system prompt if available
                    self.custom_system_prompt_text = settings.get("custom_system_prompt", "")
                    
                    # Load timeout setting
                    self.timeout = settings.get("timeout")
                    
                    # Load LlamaCpp settings
                    self.n_ctx_setting = settings.get("n_ctx", self.n_ctx_setting)
                    self.n_gpu_layers_setting = settings.get("n_gpu_layers", self.n_gpu_layers_setting)
                    self.force_cpu_only = settings.get("force_cpu_only", self.force_cpu_only)
                    
                    # Update UI
                    if hasattr(self, 'tokens_input'):
                        self.tokens_input.setText(str(self.max_tokens))
                    elif hasattr(self, 'settings_tokens_input'):
                        self.settings_tokens_input.setText(str(self.max_tokens))
                    
                    # Update LlamaCpp settings UI if available
                    if hasattr(self, 'n_ctx_input'):
                        self.n_ctx_input.setValue(self.n_ctx_setting)
                    
                    if hasattr(self, 'n_gpu_layers_input'):
                        self.n_gpu_layers_input.setValue(self.n_gpu_layers_setting)
                    
                    if hasattr(self, 'force_cpu_checkbox'):
                        self.force_cpu_checkbox.setChecked(self.force_cpu_only)
                    
                    # Find and select the current model in combo box
                    for i in range(self.model_combo.count()):
                        if self.current_model in self.model_combo.itemText(i):
                            self.model_combo.setCurrentIndex(i)
                            break
                            
                self.logger.info(f"Settings loaded: Model '{self.current_model}', Max Tokens: {self.max_tokens}, System Prompt: {'(custom)' if self.custom_system_prompt_text else '(default)'}, Timeout: {self.timeout if self.timeout else 'None (no timeout)'}")
                self.logger.info(f"LlamaCpp settings loaded: n_ctx={self.n_ctx_setting}, n_gpu_layers={self.n_gpu_layers_setting}, force_cpu_only={self.force_cpu_only}")
        except Exception as e:
            self.logger.exception("Error loading settings")
            
    def save_settings(self):
        """Save user settings to a JSON file"""
        import json
        self.logger.debug("Saving settings...")
        try:
            # Update general settings from the form
            try:
                # Update max tokens - get value from settings tab field or main UI field
                if hasattr(self, 'settings_tokens_input') and self.settings_tokens_input.text():
                    max_tokens_text = ''.join(c for c in self.settings_tokens_input.text() if c.isdigit())
                else:
                    max_tokens_text = ''.join(c for c in self.tokens_input.text() if c.isdigit())
                
                new_max_tokens = int(max_tokens_text) if max_tokens_text else self.max_tokens
                if new_max_tokens > 0:
                    self.max_tokens = new_max_tokens
                    self.logger.info(f"Updated max tokens to {self.max_tokens}")
                else:
                    self.logger.warning("Invalid max tokens value: must be > 0")
            except ValueError as e:
                self.logger.warning(f"Invalid max tokens value: {str(e)}")
            
            # Update system prompt if available in the settings form
            if hasattr(self, 'system_prompt_input'):
                self.custom_system_prompt_text = self.system_prompt_input.toPlainText().strip()
                self.logger.info(f"Updated custom system prompt: {'(custom prompt set)' if self.custom_system_prompt_text else '(using default)'}")
            
            # Update timeout
            if hasattr(self, 'timeout_input'):
                timeout_value = self.timeout_input.value()
                self.timeout = timeout_value if timeout_value > 0 else None
                self.logger.info(f"Updated timeout: {self.timeout if self.timeout else 'None (no timeout)'}")
              
            # Load existing settings first to preserve LlamaCpp settings
            settings_path = get_settings_path()  # Use our utility function
            existing_settings = {}
            try:
                if os.path.exists(settings_path):
                    with open(settings_path, "r") as f:
                        existing_settings = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load existing settings: {str(e)}")
            
            # Prepare updated settings, preserving LlamaCpp settings
            settings = {
                "model": self.current_model,
                "max_tokens": self.max_tokens,
                "custom_system_prompt": getattr(self, 'custom_system_prompt_text', ''),
                "timeout": self.timeout,
                # Preserve LlamaCpp settings
                "n_ctx": getattr(self, 'n_ctx_setting', existing_settings.get('n_ctx', 4096)),
                "n_gpu_layers": getattr(self, 'n_gpu_layers_setting', existing_settings.get('n_gpu_layers', 0)),
                "force_cpu_only": getattr(self, 'force_cpu_only', existing_settings.get('force_cpu_only', True))
            }
            
            # Save all settings to file
            with open(settings_path, "w") as f:
                json.dump(settings, f)
            self.show_status_message("Settings saved successfully", 3000)
            self.logger.info(f"Settings saved: Model='{self.current_model}', Max Tokens: {self.max_tokens}, System Prompt: {'(custom)' if getattr(self, 'custom_system_prompt_text', '') else '(default)'}, Timeout: {self.timeout if self.timeout else 'None (no timeout)'}")
            self.logger.info(f"LlamaCpp settings preserved: n_ctx={settings['n_ctx']}, n_gpu_layers={settings['n_gpu_layers']}, force_cpu_only={settings['force_cpu_only']}")
        except Exception as e:
            self.logger.exception("Error saving settings")
            self.show_status_message(f"Error saving settings: {str(e)}", 3000)
    
    def load_logs(self):
        """Load the most recent log file with newest entries at the top"""
        try:
            logs_dir = get_logs_dir()  # Use our utility function
            if not os.path.exists(logs_dir):
                self.logs_display.setPlainText("No logs directory found.")
                return
                
            log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) 
                         if f.endswith('.log')]  # Look for any log file
            
            if not log_files:
                self.logs_display.setPlainText("No log files found.")
                return
                
            recent_log = max(log_files, key=os.path.getmtime)
            
            with open(recent_log, 'r') as f:
                # Read all lines and reverse them (newest first)
                log_lines = f.readlines()
                
                # Ensure each line ends with a newline character
                log_lines = [line.rstrip('\n') + '\n' + '\n' for line in log_lines]
                
                log_lines.reverse()
                log_content = ''.join(log_lines)
                self.logs_display.setPlainText(log_content)
            if self.statusBar() is not None:    
                self.show_status_message(f"Loaded logs from {os.path.basename(recent_log)}")
        except Exception as e:
            self.logs_display.setPlainText(f"Error loading logs: {str(e)}")
    
    def closeEvent(self, a0):
        """Save settings when the application is closed"""
        try:
            # Save both general settings and LlamaCpp settings
            self.save_settings()
            self.save_llamacpp_settings()
            self.logger.info("All settings saved on application close")
        except Exception as e:
            self.logger.exception("Error saving settings on application close")
        super().closeEvent(a0)

    def toggle_web_search(self):
        """Toggle web search for the next message"""
        self.web_search_enabled = not self.web_search_enabled
        
        if self.web_search_enabled:
            self.web_button.setStyleSheet("background-color: #4CAF50; color: white;")
            if self.statusBar() is not None:
                self.show_status_message("Web search enabled for next message")
            self.logger.info("Web search enabled")
        else:
            self.web_button.setStyleSheet("")
            if self.statusBar() is not None:
                self.show_status_message("Web search disabled")
            self.logger.info("Web search disabled")

    def on_backend_changed(self, index):
        """Handle backend change"""
        if index == 0:
            self.backend = "integrated"
            self.client = self.llamacpp_client
            self.logger.info("Switched to integrated llama.cpp backend")
        else:
            self.backend = "ollama"
            self.client = self.ollama_client
            self.logger.info("Switched to Ollama backend")
        
        # Update model list based on selected backend
        self.update_model_list()

    def init_model_management_tab(self):
        """Initialize model management tab with improved Hugging Face integration"""
        model_tab = QWidget()
        layout = QVBoxLayout(model_tab)
        
        # Add explanatory text
        info_text = """<html><body>
        <h3>AI Model Options:</h3>
        <p><b>Integrated mode</b>: Download models directly through this app. Models are stored locally.</p>
        <p><b>Ollama mode</b>: Connect to models managed by Ollama.</p>
        </body></html>"""
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Backend selection
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("AI Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Integrated (llama.cpp)")
        self.backend_combo.addItem("Ollama (requires separate installation)")
        self.backend_combo.currentIndexChanged.connect(self.on_backend_changed)
        backend_layout.addWidget(self.backend_combo)
        layout.addLayout(backend_layout)
        
        # LLaMA CPP version information
        llamacpp_version = getattr(self.llamacpp_client, 'llamacpp_version', 'Unknown')
        version_label = QLabel(f"llama-cpp-python version: {llamacpp_version}")
        layout.addWidget(version_label)
        
        # Model filter and search section
        search_layout = QHBoxLayout()
        
        # Model category selection
        self.model_category = QComboBox()
        self.model_category.addItem("All Models")
        self.model_category.addItem("Predefined Models")
        self.model_category.addItem("Trending Models")
        self.model_category.addItem("Downloaded Models")
        self.model_category.currentIndexChanged.connect(self.filter_models)
        search_layout.addWidget(self.model_category)
        
        # Add search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search for models...")
        search_layout.addWidget(self.search_box)
        
        # Add search button
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search_models)
        search_layout.addWidget(search_button)
        
        layout.addLayout(search_layout)
    
        # Available models section
        layout.addWidget(QLabel("Available Models:"))
        self.available_models_list = QListWidget()
        layout.addWidget(self.available_models_list)
        
        # Model information area
        self.model_info_label = QLabel("Select a model to see details")
        self.model_info_label.setWordWrap(True)
        layout.addWidget(self.model_info_label)
        
        # Single download button
        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self.force_download_model)
        self.download_button.setToolTip("Download the selected model")
        layout.addWidget(self.download_button)

        # We're removing the progress bar for downloads as requested
        
        # Status label
        self.model_status_label = QLabel("")
        layout.addWidget(self.model_status_label)
        
        # Add refresh button
        refresh_button = QPushButton("Refresh Model List")
        refresh_button.clicked.connect(self.load_available_models)
        layout.addWidget(refresh_button)
        
        # Initialize HuggingFaceManager
        self.huggingface_manager = HuggingFaceManager()
        
        # Load available models
        self.load_available_models()
        
        # Connect selection changed signal
        self.available_models_list.itemSelectionChanged.connect(self.update_model_info)
        self.available_models_list.itemSelectionChanged.connect(self.update_download_button_state)
        
        # Add a separator line
        layout.addWidget(QLabel(""))
        layout.addWidget(QLabel("Troubleshooting:"))

        # Add debug button
        debug_button = QPushButton("Test HuggingFace API")
        debug_button.clicked.connect(self.debug_trending_models_api)
        layout.addWidget(debug_button)

        hf_version_btn = QPushButton("Check HuggingFace Version")
        hf_version_btn.clicked.connect(self.check_huggingface_version)
        layout.addWidget(hf_version_btn)
        
        # Add diagnostics button
        diagnostics_button = QPushButton("Diagnose Model Loading Issues")
        diagnostics_button.clicked.connect(self.diagnose_model_loading)
        layout.addWidget(diagnostics_button)
        
        return model_tab
    
    def update_model_list(self):
        """Get installed models from the current backend with better filename detection"""
        try:
            installed_model_ids = []
            if self.backend == "ollama":
                if not self.ollama_client.health(show_error=False):
                    self.logger.warning("Ollama not available for model list update.")
                    self.model_combo.clear()
                    if self.statusBar() is not None:
                        self.show_status_message("Ollama not available")
                    return
                installed_model_ids = self.ollama_client.list_models()
            else: # Integrated backend
                installed_model_ids = self.llamacpp_client.list_models()
            
            self.model_combo.clear()
            
            if not installed_model_ids:
                if self.backend == "ollama":
                    self.logger.warning("No models in Ollama. Please install models first.")
                    if self.statusBar() is not None:         
                        self.show_status_message("No models found in Ollama")
                else:
                    self.logger.warning("No integrated models found. Please download models or check paths.")
                if self.statusBar() is not None: 
                    self.show_status_message("No integrated models found")
                return
                    
            self.logger.info(f"Found {len(installed_model_ids)} installed models for backend '{self.backend}': {installed_model_ids}")
            
            for model_id in installed_model_ids:
                display_text = model_id  # Default display text
                
                # Try to get a more descriptive name
                if self.backend == "integrated":
                    # Use ModelManager for descriptions if available and IDs match
                    if hasattr(self.model_manager, 'available_models') and model_id in self.model_manager.available_models:
                        desc = self.model_manager.available_models[model_id].get("description", model_id)
                        display_text = f"{model_id} - {desc}"
                elif self.backend == "ollama":
                    # Use MODEL_CAPABILITIES for Ollama model descriptions
                    if model_id in MODEL_CAPABILITIES:
                        desc = MODEL_CAPABILITIES[model_id].get("description", model_id)
                        display_text = f"{model_id} - {desc}"
                
                self.model_combo.addItem(display_text, model_id) # Store the actual model_id as data
            
            # Try to reselect the current_model if it exists in the new list
            if self.current_model:
                index = self.model_combo.findData(self.current_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
                elif installed_model_ids: # If current_model not found, select the first one
                    self.model_combo.setCurrentIndex(0)
                    self.on_model_changed(self.model_combo.currentText()) # Manually trigger change
        
        except Exception as e:
            self.logger.error(f"Error loading model list: {str(e)}")
            if self.statusBar() is not None:
                self.show_status_message(f"Error loading models: {str(e)}")
    
    def load_documents(self):
        """Load documents from docs folder and add them to the list with checkboxes"""
        self.doc_list.clear()
        
        # Get docs folder path from our utility function
        docs_dir = get_docs_dir()
        self.logger.info(f"Using docs directory: {docs_dir}")
        
        # Check if directory exists and is accessible
        if os.path.isdir(docs_dir):
            try:
                # Get list of documents - CASE INSENSITIVE CHECK
                docs = [f for f in os.listdir(docs_dir) if f.lower().endswith(('.pdf', '.txt', '.md', '.docx', '.doc'))]
                
                # Add each document to the list with a checkbox
                for doc in docs:
                    item = QListWidgetItem(doc)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    self.doc_list.addItem(item)
                
                self.doc_list.itemChanged.connect(self.on_doc_selection_changed)
                self.logger.info(f"Loaded {len(docs)} documents from {docs_dir}")
            
            except Exception as e:
                self.logger.error(f"Error loading documents: {str(e)}")
                if self.statusBar() is not None:
                    self.show_status_message(f"Error loading documents: {str(e)}")
        else:
            self.logger.warning(f"Documents directory not found: {docs_dir}")
    
    def load_images(self):
        """Load images from img folder and add them to the list with checkboxes"""
        self.img_list.clear()
        
        # Get img folder path from our utility function
        img_dir = get_img_dir()
        self.logger.info(f"Using img directory: {img_dir}")
        
        # Check if directory exists and is accessible
        if os.path.isdir(img_dir):
            try:
                # Get list of images
                imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                # Add each image to the list with a checkbox
                for img in imgs:
                    item = QListWidgetItem(img)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    self.img_list.addItem(item)
                
                self.img_list.itemChanged.connect(self.on_img_selection_changed)
                self.logger.info(f"Loaded {len(imgs)} images from {img_dir}")
            
            except Exception as e:
                self.logger.error(f"Error loading images: {str(e)}")
                if self.statusBar() is not None:
                    self.show_status_message(f"Error loading images: {str(e)}")
        else:
            self.logger.warning(f"Images directory not found: {img_dir}")
    
    def on_doc_selection_changed(self, item):
        """Handle document selection change"""
        doc_name = item.text()
        # Get docs folder path from our utility function
        docs_dir = get_docs_dir()
        doc_path = os.path.join(docs_dir, doc_name)
        
        if item.checkState() == Qt.CheckState.Checked:
            if doc_path not in self.selected_docs:
                self.selected_docs.append(doc_path)
                self.logger.debug(f"Selected document: {doc_name}")
        else:
            if doc_path in self.selected_docs:
                self.selected_docs.remove(doc_path)
                self.logger.debug(f"Unselected document: {doc_name}")
    
    def on_img_selection_changed(self, item):
        """Handle image selection change"""
        img_name = item.text()
        # Get img folder path from our utility function
        img_dir = get_img_dir()
        img_path = os.path.join(img_dir, img_name)
        
        if item.checkState() == Qt.CheckState.Checked:
            if img_path not in self.selected_imgs:
                self.selected_imgs.append(img_path)
                self.logger.debug(f"Selected image: {img_name}")
        else:
            if img_path in self.selected_imgs:
                self.selected_imgs.remove(img_path)
                self.logger.debug(f"Unselected image: {img_name}")
    
    def on_model_changed(self, model_text):
        """Handle model selection change and update UI accordingly"""
        if not self.ui_initialized: # Check if UI is initialized
            return

        try:
            model_name = self.model_combo.currentData()
            if not model_name:
                if " - " in model_text:
                    model_name = model_text.split(" - ")[0].strip() # Fallback parsing
                else:
                    model_name = model_text.strip() # Fallback parsing
        
            if not model_name:
                return
                
            self.current_model = model_name
            self.logger.info(f"Changed model to: {model_name}")
            
            if self.backend == "integrated":
                try:
                    self.logger.info(f"Attempting to switch LlamaCppClient to model: {model_name}")
                    if not self.llamacpp_client.switch_model(model_name): # *** UNCOMMENT AND ENSURE THIS LINE IS ACTIVE ***
                        self.logger.error(f"Failed to switch to model {model_name} via LlamaCppClient.")
                        QMessageBox.warning(self, "Model Load Error", f"Could not load model: {model_name}")
                    else:
                        self.logger.info(f"Successfully switched LlamaCppClient to model: {model_name}")
                except Exception as e:
                    self.logger.exception(f"Error switching model {model_name}")
                    QMessageBox.critical(self, "Model Load Error", f"Error loading model {model_name}: {str(e)}")
        
            self.update_ui_for_model_capabilities()
            self.logger.info(f"Model changed to {self.current_model}, UI updated for capabilities.") # Log moved after UI update
            
        except Exception as e:
            self.logger.error(f"Error in on_model_changed: {str(e)}")
        
    def update_ui_for_model_capabilities(self):
        """Update UI elements based on current model capabilities"""
        if not self.ui_initialized: # Check if UI is initialized
            return

        # Get capabilities for current model
        capabilities = MODEL_CAPABILITIES.get(self.current_model, {
            "doc": False,
            "img": False,
            "ocr": False,
            "web": False,
            "rag": False
        })
        
        # Enable/disable document list items
        can_use_docs = capabilities.get("doc", False)
        for i in range(self.doc_list.count()):
            item = self.doc_list.item(i)
            if item is None:
                continue
                  # If model can't use docs, uncheck and disable
            if not can_use_docs:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(Qt.ItemFlag(item.flags() & ~Qt.ItemFlag.ItemIsEnabled))
            else:
                item.setFlags(Qt.ItemFlag(item.flags() | Qt.ItemFlag.ItemIsEnabled))

        # Enable/disable image list items
        can_use_imgs = capabilities.get("img", False) or capabilities.get("ocr", False)
        for i in range(self.img_list.count()):
            item = self.img_list.item(i)
            if item is None:
                continue
                  # If model can't use images, uncheck and disable
            if not can_use_imgs:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(Qt.ItemFlag(item.flags() & ~Qt.ItemFlag.ItemIsEnabled))
            else:
                item.setFlags(Qt.ItemFlag(item.flags() | Qt.ItemFlag.ItemIsEnabled))
    
    def check_ollama(self):
        """Check if Ollama is available, but only if using Ollama backend"""
        # Skip check if using integrated backend
        if self.backend != "ollama":
            return True
            
        # Only check Ollama when using that backend
        if not self.ollama_client.health():
            QMessageBox.warning(self, "Ollama Not Running", 
                               "Could not connect to Ollama. Please ensure it's running.")
            if self.statusBar() is not None:
                self.show_status_message("Error: Ollama not running")
            return False
        else:
            if self.statusBar() is not None:
                self.show_status_message("Connected to Ollama")
            return True
    
    def send_message(self):
        if not self.message_input.text().strip():
            return
        
        message = self.message_input.text()
        self.logger.info(f"Send message initiated by user. Message: '{message[:20]}...'")
        self.add_to_chat("User", message)
        self.message_input.clear()
        self.send_button.setEnabled(False)
        self.progress.show()
        self.history.append({"role": "user", "content": message})
        
        max_tokens = self.max_tokens
        self.logger.info(f"Using max_tokens: {max_tokens}")
        
        # Store the current time for performance metrics
        self.start_time = time.time()
        
        # Get the client from model_manager to ensure it's the same one that loaded the model
        client = self.model_manager.get_active_client()
          # Get custom system prompt based on current model
        custom_system_prompt = self.get_system_prompt_for_model(self.current_model)
        
        # Format history for the model
        formatted_history = []
        # Use only the last 6 messages to avoid context length issues
        for entry in self.history[-6:]:
            formatted_history.append({
                "role": entry["role"],
                "content": entry["content"]
            })
        
        # Prepare options with system prompt and history
        options = {
            "system_message": custom_system_prompt,
            "history": formatted_history,  # Add conversation history
            "temperature": 0.7,            # Lower temperature = less hallucination
            "top_p": 0.8,                 # Lower top_p = more focused responses
            "repetition_penalty": 1.15     # Prevent repetitive outputs
        }
          # Use timeout from settings if available
        timeout_value = getattr(self, 'timeout', None)
        self.logger.info(f"Using timeout: {timeout_value if timeout_value else 'None (no timeout)'}")
        
        self.generation_thread = GenerationThread(
            client=client,
            model=self.current_model,
            prompt=message,
            max_tokens=max_tokens,
            timeout=timeout_value,  # Use the timeout from settings
            options=options  # Pass options with our custom system prompt and history
        )
        
        self.generation_thread.finished.connect(self._handle_response)
        self.generation_thread.start()
        # REMOVE THIS LINE: self.thread.start()
    
    def build_context_from_selections(self):
        """Build context text from selected documents and images"""
        context_parts = []
        
        # Add document contexts
        for doc_path in self.selected_docs:
            try:
                doc_name = os.path.basename(doc_path)
                doc_text = process_document(doc_path)
                context_parts.append(f"[Document: {doc_name}]\n{doc_text}")
            except Exception as e:
                self.logger.error(f"Error processing document {doc_path}: {str(e)}")
                context_parts.append(f"[Error processing document: {doc_path}]")
        
        # Add OCR from images if the model supports OCR but not direct image analysis
        if MODEL_CAPABILITIES.get(self.current_model, {}).get("ocr", False) and not MODEL_CAPABILITIES.get(self.current_model, {}).get("img", False):
            for img_path in self.selected_imgs:
                try:
                    img_name = os.path.basename(img_path)
                    img_text = process_image(img_path)
                    context_parts.append(f"[Image OCR: {img_name}]\n{img_text}")
                except Exception as e:
                    self.logger.error(f"Error processing image {img_path}: {str(e)}")
                    context_parts.append(f"[Error processing image: {img_path}]")
        
        return "\n\n".join(context_parts)
    
    def _format_chat_history(self):
        """Format chat history for the prompt"""
        history_text = ""
        for entry in self.history[-10:]:  # Use last 10 messages
            if entry["role"] == "user":
                history_text += f"User: {entry['content']}\n"
            elif entry["role"] == "assistant":
                history_text += f"Assistant: {entry['content']}\n"
        return history_text
    def _clean_hallucinated_content(self, response_text):
        """Remove hallucinated content like fake conversations and prefixes"""
        # Original cleaning: Remove prefixes
        response_text = re.sub(r'^(and|Ladbon AI:|AI:|Assistant:|User:)\s*', '', response_text.strip())
        
        # Remove any lines starting with "User:" or "Human:" (hallucinated questions)
        response_text = re.sub(r'(?m)^User:.*?(?:\n|$)', '', response_text)
        response_text = re.sub(r'(?m)^Human:.*?(?:\n|$)', '', response_text)
        
        # Remove any lines starting with "Assistant:" 
        response_text = re.sub(r'(?m)^Assistant:.*?(?:\n|$)', '', response_text)
        
        # Remove hashtag formatting the model sometimes adds
        response_text = re.sub(r'\s+#\w+(\s+#\w+)*\s*$', '', response_text)
        
        # Remove any sections that look like hallucinated conversations
        response_text = re.sub(r'User: .*?Assistant: .*?(\n|$)', '', response_text)
        
        return response_text.strip()
        
    def _handle_response(self, response_text): 
        self.logger.info(f"Received response from AI thread. Length: {len(response_text)}")
        self.progress.hide()
        self.send_button.setEnabled(True)
        
        elapsed = time.time() - self.start_time if hasattr(self, "start_time") else 0
        
        # Clean response text to remove hallucinations
        cleaned_response = self._clean_hallucinated_content(response_text.strip())
        
        # Process the response for better formatting
        formatted_response = self.format_ai_response(cleaned_response)
        
        # Display the formatted response
        self.add_to_chat("AI", formatted_response)
        
        # Add performance metrics
        tokens_estimated = len(response_text) / 4
        tokens_per_second = tokens_estimated / elapsed if elapsed > 0 else 0
        metrics = f"[Generated in {elapsed:.2f}s, ~{int(tokens_per_second)} tokens/s]"
        self.chat_display.append(f"<div style='color: gray; font-size: 8pt; margin-top:5px; text-align: right;'>{metrics}</div>")
        
        # Add to history
        self.history.append({"role": "assistant", "content": response_text})
        
        # Trim history if needed
        if len(self.history) > 20:
            self.history = self.history[-20:]

        self.chat_display.append("\n")
        self.chat_display.ensureCursorVisible()   

    def filter_models(self):
        """Filter models based on selected category"""
        self.available_models_list.clear()
    
        if self.backend != "integrated":
            self.available_models_list.addItem("Use 'ollama pull model_name' to download models with Ollama")
            return
            
        category = self.model_category.currentText()
    
        try:
            # Get all models
            all_models = self.huggingface_manager.list_available_models(include_trending=True)
            
            # Filter based on category
            filtered_models = {}
            
            if category == "All Models":
                filtered_models = all_models
            elif category == "Predefined Models":
                filtered_models = {k: v for k, v in all_models.items() if v.get("is_predefined")}
            elif category == "Trending Models":
                filtered_models = {k: v for k, v in all_models.items() if v.get("is_trending")}
            elif category == "Downloaded Models":
                filtered_models = {k: v for k, v in all_models.items() if v.get("is_downloaded")}
            
            # Add models to list
            self._populate_model_list(filtered_models)
            self.logger.info(f"Filtered to {len(filtered_models)} {category}")
            
        except Exception as e:
            self.logger.error(f"Error filtering models: {str(e)}")
            self.model_status_label.setText(f"Error: {str(e)}")

    def search_models(self):
        """Search for models based on search box input"""
        if self.backend != "integrated":
            return
            
        query = self.search_box.text().strip()
        if not query or len(query) < 2:
            self.model_status_label.setText("Please enter a search term (min 2 characters)")
            return
            
        self.model_status_label.setText(f"Searching for '{query}'...")
        self.available_models_list.clear()
        
        try:
            # Perform the search
            search_results = self.huggingface_manager.search_models(query)
            
            if not search_results:
                self.model_status_label.setText(f"No models found matching '{query}'")
                return
                
            # Add search results to list
            self._populate_model_list(search_results)
            
            self.model_status_label.setText(f"Found {len(search_results)} models matching '{query}'")
            self.model_category.setCurrentText("All Models")  # Reset category filter
            
        except Exception as e:
            self.logger.error(f"Error searching models: {str(e)}")
            self.model_status_label.setText(f"Search error: {str(e)}")

    def _populate_model_list(self, models_dict):
        """Populate the model list widget from a dictionary of models"""
        for model_id, info in models_dict.items():
            # Create list item with model info
            description = info.get("description", "").split("\n")[0][:50]  # First line, truncated
            if len(description) >= 50:
                description += "..."
                
            # Format item text based on what metadata is available
            item_text = f"{model_id}"
            
            if "downloads" in info and info["downloads"]:
                item_text += f" ({info['downloads']} downloads)"
                
            item_text += f" - {description}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, model_id)

            # Style based on model type - avoid icon theme which may not be supported
            if info.get("is_trending"):
                # Just add a star symbol instead of using an icon
                item.setText("★ " + item_text) 
            
            if info.get("is_predefined"):
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                
            # Mark downloaded models with green background
            if info.get("is_downloaded"):
                item.setBackground(Qt.GlobalColor.green)
                size_mb = info.get("size_mb", 0)
                item.setToolTip(f"Downloaded ({size_mb:.1f} MB)")
            
            self.available_models_list.addItem(item)
        
        # Sort the list to make models easier to find
        self.available_models_list.sortItems()

    def load_available_models(self):
        """Load available models with metadata from HuggingFaceManager"""
        # Save current selection if any
        selected_model_id = None
        selected_items = self.available_models_list.selectedItems()
        if selected_items and len(selected_items) > 0:
            selected_model_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        # Clear and reload the list
        self.available_models_list.clear()
        self.model_status_label.setText("Loading models...")
        
        if self.backend == "integrated":
            try:
                # Get models from HuggingFaceManager (including trending)
                models = self.huggingface_manager.list_available_models(include_trending=True)
                
                # Add models to list
                self._populate_model_list(models)
                
                self.logger.info(f"Loaded {len(models)} available models")
                self.model_status_label.setText(f"Loaded {len(models)} models")
                
                # Restore selection if possible
                if selected_model_id:
                    for i in range(self.available_models_list.count()):
                        item = self.available_models_list.item(i)
                        if item and item.data(Qt.ItemDataRole.UserRole) == selected_model_id:
                            self.available_models_list.setCurrentItem(item)
                            break
                
            except Exception as e:
                self.logger.error(f"Error loading available models: {str(e)}")
                self.model_status_label.setText(f"Error: {str(e)}")
        else:
            self.available_models_list.addItem("Use 'ollama pull model_name' to download models with Ollama")
        
        # Update download button state
        self.update_download_button_state()

    def update_download_button_state(self):
        """Update download button state based on selection and download status"""
        if self.backend != "integrated":
            self.download_button.setEnabled(False)
            self.download_button.setText("Use Ollama Pull Command")
            return
            
        selected_items = self.available_models_list.selectedItems()
        
        # Check if there's a selection
        if selected_items and len(selected_items) > 0:
            self.download_button.setEnabled(True)
            
            # Get the model ID and check if it's already downloaded
            model_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
            
            if hasattr(self, 'huggingface_manager'):
                all_models = self.huggingface_manager.list_available_models(include_trending=True)
                if model_id in all_models and all_models[model_id].get('is_downloaded', False):
                    # Model is already downloaded, rename button to "Redownload"
                    self.download_button.setText("Redownload")
                else:
                    # Model is not yet downloaded
                    self.download_button.setText("Download")
            else:
                # Default state if huggingface_manager not ready
                self.download_button.setText("Download")
        else:
            # No selection
            self.download_button.setEnabled(False)
            self.download_button.setText("Download")

    def update_model_info(self):
        """Update model information when selection changes"""
        selected_items = self.available_models_list.selectedItems()
        if not selected_items:
            self.model_info_label.setText("Select a model to see details")
            # Also update download button state
            self.update_download_button_state()
            return
        
        item = selected_items[0]
        model_id = item.data(Qt.ItemDataRole.UserRole)
        
        if self.backend == "integrated":
            # Get model information from template
            models = self.huggingface_manager.list_available_models()
            if model_id in models:
                info = models[model_id]
                
                # Build info text
                info_text = f"<b>{model_id}</b><br>"
                info_text += f"Repository: {info.get('repo_id', 'Unknown')}<br>"
                info_text += f"Description: {info.get('description', 'Unknown')}<br>"
                
                if info.get("is_downloaded"):
                    size_mb = info.get("size_mb", 0)
                    info_text += f"<span style='color:green;'>Downloaded</span> ({size_mb:.1f} MB)<br>"
                    info_text += f"Path: {info.get('local_path', 'Unknown')}"
                    self.model_info_label.setText(info_text)
                else:
                    # Try to get size information from Hugging Face
                    self.model_info_label.setText("Querying model details from Hugging Face...")
                    
                    # Define the get_info_thread function here
                    def get_info_thread():
                        try:
                            model_info = self.huggingface_manager.get_model_info(model_id)
                            from PyQt5.QtCore import QTimer
                            QTimer.singleShot(0, lambda: self.update_model_info_ui(model_id, model_info))
                        except Exception as e:
                            from PyQt5.QtCore import QTimer
                            self.logger.error(f"Error getting model info: {str(e)}")
                            QTimer.singleShot(0, lambda: self.model_info_label.setText(f"Error: {str(e)}"))
                    
                    # Now start the thread
                    import threading
                    thread = threading.Thread(target=get_info_thread)
                    thread.daemon = True
                    thread.start()
            else:
                self.model_info_label.setText(f"No detailed information available for {model_id}")
        else:
            self.model_info_label.setText("Model information not available in Ollama mode")

    # Removed download_model method - using force_download_model directly

    def force_download_model(self):
        """Force download selected model"""
        self.logger.info("Force download model requested")
        
        # Get selected model ID
        selected_items = self.available_models_list.selectedItems()
        if not selected_items or len(selected_items) == 0:
            self.logger.warning("No model selected for download")
            QMessageBox.warning(self, "No Model Selected", "Please select a model to download.")
            return
        
        item = selected_items[0]
        model_id = item.data(Qt.ItemDataRole.UserRole)
        
        if not model_id:
            self.logger.warning("Selected item has no model ID")
            QMessageBox.warning(self, "Invalid Selection", "The selected item does not have a valid model ID.")
            return
        
        self.logger.info(f"Starting download for model: {model_id}")
        
        # Check if HuggingFaceManager is initialized
        if not hasattr(self, 'huggingface_manager'):
            from utils.huggingface_manager import HuggingFaceManager
            from utils.data_paths import get_models_dir
            models_dir = get_models_dir()
            self.huggingface_manager = HuggingFaceManager(models_dir=models_dir)
            self.logger.info(f"Initialized HuggingFaceManager with models_dir: {models_dir}")
        
        # Check if model info exists
        models = self.huggingface_manager.list_available_models(include_trending=True)
        if model_id not in models:
            self.logger.warning(f"Model {model_id} not found in available models")
            QMessageBox.warning(self, "Model Not Found", f"Could not find model {model_id} in available models.")
            return
        
        # Prepare UI for download
        self.download_button.setEnabled(False)  # Disable to prevent multiple downloads
        self.download_button.setText("Downloading...")  # Visual feedback on button
        self.model_status_label.setText(f"Starting download of {model_id}...")
        if self.statusBar() is not None:
            self.show_status_message(f"Downloading {model_id}...")
        
        # Create a thread to avoid blocking UI
        # Store thread and worker as instance variables to prevent garbage collection
        if not hasattr(self, 'download_thread'):
            self.download_thread = None
        if not hasattr(self, 'download_worker'):
            self.download_worker = None
            
        # Clean up any existing thread before creating a new one
        if self.download_thread is not None:
            if self.download_thread.isRunning():
                self.logger.warning("Previous download thread still running, attempting to stop it")
                self.download_thread.quit()
                self.download_thread.wait(3000)  # Wait up to 3 seconds for thread to finish
            
            self.download_thread = None
            self.download_worker = None
        
        # Create a new thread
        self.download_thread = QThread()
        
        # Define worker class to run download in background
        class DownloadWorker(QObject):
            finished = pyqtSignal(bool, str)
            progress = pyqtSignal(int)
            log = pyqtSignal(str)
            
            def __init__(self, model_id, huggingface_manager):
                super().__init__()
                self.model_id = model_id
                self.huggingface_manager = huggingface_manager
                
            def run(self):
                try:
                    self.log.emit(f"Download thread started for {self.model_id}")
                    
                    # Show starting message
                    self.progress.emit(0) # Signal 0% progress
                    
                    # Create a progress callback function that updates status text and logs at key intervals
                    def progress_callback(percentage):
                        self.progress.emit(percentage)  # Update UI via signal
                    
                    # Start the download with detailed logging
                    self.log.emit(f"INITIATING DOWNLOAD FOR MODEL: {self.model_id}")
                    success, result = self.huggingface_manager.simple_download_model(
                        model_id=self.model_id,
                        progress_callback=progress_callback
                    )
                    
                    self.log.emit(f"Download thread completed for {self.model_id}, success={success}")
                    self.finished.emit(success, result)
                    
                except Exception as e:
                    import traceback
                    self.log.emit(f"Error in download thread: {str(e)}")
                    self.log.emit(traceback.format_exc())
                    # Signal completion with error
                    self.finished.emit(False, f"Error: {str(e)}")
        
        # Create worker and connect signals
        self.download_worker = DownloadWorker(model_id, self.huggingface_manager)
        self.download_worker.moveToThread(self.download_thread)
        
        # Connect thread start to worker's run method
        self.download_thread.started.connect(self.download_worker.run)
        
        # Connect log messages
        self.download_worker.log.connect(lambda msg: self.logger.info(msg))
        
        # Connect progress updates
        def handle_progress(percentage):
            # Update status label for all progress updates
            self.model_status_label.setText(f"Downloading: {percentage}%")
            
            # Only update status bar at 10% intervals for cleaner UI
            if percentage == 0 or percentage == 100 or percentage % 10 == 0:
                self.show_status_message(f"Downloading: {percentage}%")
                # Log visibly at these points
                self.logger.info(f"Download progress for {model_id}: {percentage}%")
            
            # When download reaches 100%, ensure immediate UI update
            if percentage == 100:
                self.logger.info(f"Download reached 100% for model {model_id}")
                # Reset UI state immediately after download is finished
                self.download_button.setText("Redownload")
                self.download_button.setEnabled(True)
                
        self.download_worker.progress.connect(handle_progress)
        
        # When thread is done, process results
        def process_result(success, result):
            if success:
                output_path = result
                self._handle_download_complete(True, result, model_id)
            else:
                error_message = result
                self._handle_download_complete(False, error_message, model_id)
            
            # Explicitly clean up thread references after completion
            self.download_thread = None
            self.download_worker = None
        
        self.download_worker.finished.connect(process_result)
        
        # Handle thread finish (connect these AFTER process_result to ensure they run last)
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_thread.finished.connect(lambda: self.logger.info("Download thread finished and resources released"))
        
        # Start the thread
        self.download_thread.start()
        self.logger.info(f"Download thread started for {model_id}")
    
    def debug_model_status(self):
        """Print detailed debugging info about model status"""
        print("\n=== MODEL STATUS DEBUGGING ===")
        
        # Check model directory
        models_dir = self.model_manager.models_dir
        print(f"Models directory: {models_dir}")
        
        if os.path.exists(models_dir):
            print("Directory exists")
            files = os.listdir(models_dir)
            print(f"Files in directory: {files}")
            
            # Check each file
            for filename in files:
                if filename.endswith('.gguf'):
                    full_path = os.path.join(models_dir, filename)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"Model combo items: {[self.model_combo.itemText(i) for i in range(self.model_combo.count())]}")
        print("=== END DEBUGGING ===\n")

    def restart_application(self):
        """Restart the entire application to ensure clean state"""
        self.logger.info("Restarting application after model download...")
        
        # Save any important state
        try:
            # Save settings before exiting
            self.save_settings()
            self.save_llamacpp_settings()
            
            # Get the current executable path
            import sys
            import os
            import subprocess
            
            python = sys.executable
            script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui_app.py")
            
            # Launch a new instance
            self.logger.info(f"Launching new instance: {python} {script}")
            subprocess.Popen([python, script])
            
            # Exit the current instance after a slight delay
            def exit_app():
                QApplication.quit()
            
            QTimer.singleShot(500, exit_app)
            
        except Exception as e:
            self.logger.error(f"Failed to restart application: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to restart: {str(e)}")
    
    def debug_trending_models_api(self):
        """Debug trending models API issues"""
        from PyQt5.QtWidgets import QMessageBox
        import requests
        import json
        import textwrap
        
        self.model_status_label.setText("Testing Hugging Face API...")
        self.logger.info("Testing Hugging Face API...")
        
        try:
            # Test a simple API call
            url = "https://huggingface.co/api/models?sort=downloads&direction=-1&filter=gguf&limit=5" 
            self.logger.info(f"Testing URL: {url}")
            
            response = requests.get(url)
            self.logger.info(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                models = response.json()
                self.logger.info(f"Found {len(models)} models")
                
                if models:
                    # Log first model details
                    first_model = models[0]
                    
                    # Show in UI
                    self.model_status_label.setText(f"API working! Found {len(models)} models")
                    
                    # Show model details in message box
                    model_info = f"Model ID: {first_model.get('id')}\n"
                    model_info += f"Downloads: {first_model.get('downloads')}\n"
                    model_info += f"Likes: {first_model.get('likes')}\n"
                    
                    # Available fields
                    model_info += f"\nAvailable fields: {', '.join(first_model.keys())}\n\n"
                    
                    # Download path
                    if 'id' in first_model:
                        repo_id = first_model['id']
                        model_info += f"Would download from repo: {repo_id}\n"
                        
                        # Show where it would be saved
                        local_dir = os.path.abspath(self.huggingface_manager.models_dir)
                        model_info += f"Would save to: {local_dir}\n"
                    
                    QMessageBox.information(self, "API Test Results", model_info)
                else:
                    self.model_status_label.setText("API returned 0 models")
            else:
                self.model_status_label.setText(f"API Error: {response.status_code}")
                QMessageBox.warning(self, "API Error", 
                                f"Server returned status code: {response.status_code}")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"API Test Error: {str(e)}\n{error_details}")
            self.model_status_label.setText(f"API Test Error: {str(e)}")
           
            QMessageBox.critical(self, "API Error", f"Error: {str(e)}\n\n{error_details}")
    
    def check_huggingface_version(self):
        """Check HuggingFace Hub version and capabilities"""
        from PyQt5.QtWidgets import QMessageBox
        import inspect
        
        try:
            # Get version info
            import huggingface_hub
            version = huggingface_hub.__version__
            
            # Check hf_hub_download parameters
            from huggingface_hub import hf_hub_download
            params = inspect.signature(hf_hub_download).parameters
            param_names = list(params.keys())
            
            # Build info message
            info = f"HuggingFace Hub Version: {version}\n\n"
            info += "hf_hub_download parameters:\n"
            info += "\n".join([f"- {p}" for p in param_names])
            
            # Check if important parameters are available
            has_progress = 'progress_callback' in param_names
            has_force = 'force_download' in param_names
            has_resume = 'resume_download' in param_names
            
            info += f"\n\nSupports progress_callback: {has_progress}"
            info += f"\nSupports force_download: {has_force}"
            info += f"\nSupports resume_download: {has_resume}"
            
            # Show the information
            QMessageBox.information(self, "HuggingFace Hub Info", info)
            
            # Log this information
            self.logger.info(f"HuggingFace Hub Version: {version}")
            self.logger.info(f"hf_hub_download parameters: {', '.join(param_names)}")
            
            # If missing important parameters, suggest an upgrade
            if not has_progress or not has_force or not has_resume:
                upgrade_msg = ("Your version of huggingface_hub is missing some useful features.\n\n"
                              "Consider upgrading with:\n"
                              "pip install --upgrade huggingface_hub")
                QMessageBox.warning(self, "Consider Upgrading", upgrade_msg)
    
        except Exception as e:
            self.logger.error(f"Error checking HuggingFace version: {str(e)}")
            QMessageBox.critical(self, "Error", f"Could not check HuggingFace version: {str(e)}")
    
    def diagnose_model_loading(self):
        """Run diagnostics on model loading capabilities"""
        self.logger.info("Starting model loading diagnostics...")
        
        if self.backend == "ollama":
            QMessageBox.information(self, "Ollama Diagnostics", "Olloma models are managed by Olloma service.")
            return
        
        # Get selected model or use current model
        model_id = self.current_model
        selected_items = self.available_models_list.selectedItems()
        if selected_items:
            model_id = selected_items[0].data(Qt.ItemDataRole.UserRole) or model_id
        
        if not model_id:
            QMessageBox.warning(self, "No Model", "No model selected for diagnostics.")
            return
            
        try:
            # Basic diagnostics
            results = {
                "Model ID": model_id,
                "Backend": self.backend,
                "LlamaCpp Health": self.llamacpp_client.health(),
                "Available Models": len(self.llamacpp_client.list_models()),
                "Current Loaded": self.llamacpp_client.model_path or "None"
            }
            
            # Format results
            report = "\n".join([f"{k}: {v}" for k, v in results.items()])
            
            # Show in dialog
            QMessageBox.information(self, "Model Diagnostics", f"Diagnostics Results:\n\n{report}")
            
            # Also log it
            self.logger.info(f"Model Loading Diagnostics:\n{report}")
            
        except Exception as e:
            self.logger.error(f"Error in diagnostics: {str(e)}")
            QMessageBox.warning(self, "Diagnostic Error", f"Error running diagnostics: {str(e)}")

    def restart_llamacpp(self):
        """Restart the llama-cpp backend to detect new models"""
        self.logger.info("Restarting llama-cpp backend to detect new models...")
        
        try:
            # Remember current model
            previous_model = self.current_model
            
            # First, make sure any loaded model is unloaded
            if hasattr(self.llamacpp_client, 'loaded_model') and self.llamacpp_client.loaded_model:
                self.logger.info("Unloading current model...")
                model_path = self.llamacpp_client.model_path
                # Release model and memory
                del self.llamacpp_client.loaded_model
                self.llamacpp_client.loaded_model = None
                self.llamacpp_client.model_path = None
                gc.collect()  # Force garbage collection
                self.logger.info(f"Successfully unloaded model from {model_path}")
            
            # Get current settings before reinitializing
            current_ctx = getattr(self.llamacpp_client, 'n_ctx', 4096)
            current_gpu_layers = getattr(self.llamacpp_client, 'n_gpu_layers', 0)
            
            # Create a new instance of LlamaCppClient
            self.logger.info(f"Creating new LlamaCppClient with n_ctx={current_ctx}, n_gpu_layers={current_gpu_layers}")
            self.llamacpp_client = LlamaCppClient(n_ctx=current_ctx, n_gpu_layers=current_gpu_layers)
            
            # Log the models path being used
            from utils.data_paths import get_models_dir
            models_dir = get_models_dir()
            self.logger.info(f"Newly restarted LlamaCppClient will search for models in: {models_dir}")
            
            # Update the client reference in the model manager
            if hasattr(self, 'model_manager'):
                self.model_manager.llamacpp_client = self.llamacpp_client
                self.logger.info("Updated model_manager's LlamaCppClient reference")
            
            # Update the client reference if needed
            if self.backend == "integrated":
                self.client = self.llamacpp_client
                self.logger.info("Updated integrated backend client reference")
            
            # Refresh model lists and UI
            self.update_model_list()
            
            # Auto-load the current model after restart with extra verification
            if previous_model:
                self.logger.info(f"Attempting to reload model '{previous_model}' after restart")
                try:
                    # Try up to 3 times to ensure model loads properly
                    max_attempts = 3
                    for attempt in range(1, max_attempts + 1):
                        self.logger.info(f"Model reload attempt {attempt}/{max_attempts}")
                        
                        # Force clean previous attempts
                        if attempt > 1 and hasattr(self.llamacpp_client, 'loaded_model'):
                            if self.llamacpp_client.loaded_model:
                                del self.llamacpp_client.loaded_model
                                self.llamacpp_client.loaded_model = None
                            gc.collect()
                            time.sleep(0.5)  # Brief pause between attempts
                        
                        if self.llamacpp_client.switch_model(previous_model):
                            self.logger.info(f"Successfully reloaded model '{previous_model}' after restart")
                            self.current_model = previous_model
                            
                            # Verify model is actually loaded
                            is_healthy = self.llamacpp_client.health()
                            self.logger.info(f"Health check after reload: {is_healthy}")
                            if is_healthy:
                                break
                            else:
                                self.logger.warning("Model reported loaded but failed health check")
                        else:
                            self.logger.warning(f"Failed to reload model '{previous_model}' - attempt {attempt}")
                            
                    # If we couldn't load the model after multiple attempts, log an error
                    if not getattr(self.llamacpp_client, 'loaded_model', None):
                        self.logger.error(f"Could not reload model '{previous_model}' after multiple attempts")
                except Exception as load_error:
                    self.logger.error(f"Error during model reload: {str(load_error)}")
            
            self.logger.info("LlamaCpp backend restarted successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to restart llama-cpp backend: {str(e)}")
            return False

    def _handle_download_complete(self, success, message, model_id):
        """Handle completion of model download with proper UI updates"""
        # Button state should already be updated in the download thread
        self.logger.info(f"Processing download completion for {model_id}, success={success}")
            
        if success:
            self.logger.info(f"Download complete for {model_id}: {message}")
            
            if self.statusBar() is not None:
                self.show_status_message(f"Download complete: {model_id}")
                
            if hasattr(self, 'model_status_label'):
                # Clear the "Downloading..." message
                self.model_status_label.setText(f"Download complete: {model_id}")
                
                # Set a timer to clear the status after 5 seconds
                QTimer.singleShot(5000, lambda: self.model_status_label.setText(""))
            
            # Show success message
            QMessageBox.information(self, "Download Complete", message)
            
            # CRITICAL: Before refreshing UI, restart llama-cpp to detect new models
            self.restart_llamacpp()
            
            # CRITICAL: Refresh the model lists to ensure the new model is available
            self.logger.info("Refreshing model lists after download")
            self.load_available_models()
            
            # Update the model list in the top right combobox 
            self.logger.info("Updating model dropdown in UI")
            self.update_model_list()
            
            # Auto-select the downloaded model in the available models list
            self.logger.info(f"Auto-selecting model {model_id} in available models list")
            self.select_model_by_id(model_id)
            
            # CRITICAL: Auto-mount the model by selecting it in the model combobox
            if hasattr(self, 'model_combo'):
                self.logger.info(f"Attempting to auto-mount downloaded model {model_id}")
                
                # Get the simplified model ID that would be used by llamacpp_client
                simplified_model_id = self._get_simplified_model_id(model_id)
                
                # First try exact match
                index = self.model_combo.findData(model_id)
                # Then try with simplified ID
                if index < 0:
                    index = self.model_combo.findData(simplified_model_id)
                # Try alternative ways to find the model by text 
                if index < 0:
                    for i in range(self.model_combo.count()):
                        item_text = self.model_combo.itemText(i)
                        if model_id.lower() in item_text.lower() or simplified_model_id.lower() in item_text.lower():
                            index = i
                            break
                
                if index >= 0:
                    self.logger.info(f"Found model {model_id} at index {index}, auto-mounting...")
                    self.model_combo.setCurrentIndex(index)
                    # This will trigger on_model_changed
                else:
                    # If we couldn't find the model in the combobox, try refreshing again
                    self.logger.warning(f"Could not find model {model_id} in combobox, refreshing model list again")
                    # Force another update of the model list
                    self.update_model_list()
                    
                    # Try to find the model again
                    index = self.model_combo.findData(model_id)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                        self.logger.info(f"Found model {model_id} after refresh, auto-mounting...")
                    else:
                        self.logger.warning(f"Still could not find model {model_id} in combobox after refresh")
            else:
                self.logger.warning("No model combobox found, cannot auto-mount model")
            
            # Make sure the download button state is properly updated
            self.update_download_button_state()
        else:
            # Handle failure case
            self.logger.error(f"Download failed for {model_id}: {message}")
            
            if self.statusBar() is not None:
                self.show_status_message(f"Download failed: {model_id}")
                
            if hasattr(self, 'model_status_label'):
                self.model_status_label.setText(f"Download failed: {message}")
                
                # Clear the error message after 10 seconds
                QTimer.singleShot(10000, lambda: self.model_status_label.setText(""))
                
            QMessageBox.critical(self, "Download Failed", f"Failed to download {model_id}:\n{message}")
            
            # Still refresh the model list just in case partial download happened
            self.load_available_models()
            self.update_model_list()
            self.update_download_button_state()
            
    def select_model_by_id(self, model_id):
        """Select a model in the available models list by its ID"""
        if not model_id:
            return
            
        self.logger.info(f"Auto-selecting model {model_id} in the available models list")
        
        # Find and select the model in the list
        for i in range(self.available_models_list.count()):
            item = self.available_models_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == model_id:
                self.available_models_list.setCurrentItem(item)
                # Ensure the item is visible
                self.available_models_list.scrollToItem(item)
                return True
                
        self.logger.warning(f"Could not find model {model_id} in the available models list")
        return False

    def _map_filename_to_model_id(self, filename):
        """Map a filename to a model ID for display in the model selector"""
        lower_name = filename.lower()
        
        # Common naming patterns - FIXED VERSION
        if "llama-2" in lower_name:
            return "llama2-7b"
        elif "llama2" in lower_name:
            return "llama2-7b"
        elif "tiny" in lower_name and "llama" in lower_name:
            return "tinyllama"
        elif "mistral" in lower_name:
            if "small" in lower_name:
                return "mistral-small"
            else:
                return "mistral-7b"
        elif "phi" in lower_name:
            if "mini" in lower_name:
                return "phi3-mini"
            else:
                return "phi3"
        elif "medgemma" in lower_name:
            return "medgemma"
        elif "qwen" in lower_name:
            return "qwen"
        else:
            # Use the first part of filename as model ID
            parts = filename.split('-')[0].split('.')
            return parts[0].lower()
            
    def _format_code_block(self, match):
        """Format code blocks with syntax highlighting"""
        language = match.group(1).strip() or "plaintext"
        code = match.group(2)
        
        # Clean up indentation for consistent display
        lines = code.split('\n')
        # Find minimum indentation (ignoring empty lines)
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            min_indent = min((len(line) - len(line.lstrip(' '))) for line in non_empty_lines)
            # Remove that amount of indentation from each line
            cleaned_code = '\n'.join(line[min_indent:] if line.strip() else line for line in lines)
        else:
            cleaned_code = code
        
        # Add proper escaping for HTML
        escaped_code = (cleaned_code.replace('&', '&amp;')
                                .replace('<', '&lt;')
                                .replace('>', '&gt;'))
        
        # Use a wrapping div to ensure full background coverage
        return f"""
        <div style="background-color: #1e1e2e; color: #f8f8f2; padding: 16px 20px; 
            border-radius: 5px; margin: 16px 0px; overflow: hidden; width: 100%;">
            <div style="color: #bd93f9; font-size: 0.9em; margin-bottom: 8px; border-bottom: 1px solid #444; padding-bottom: 5px;">
                {language}
            </div>
            <pre style="margin: 0; white-space: pre; tab-size: 4; font-family: 'Consolas', 'Courier New', monospace; line-height: 1.4; background-color: transparent; color: inherit; overflow-x: auto;"><code>{escaped_code}</code></pre>
        </div>
        """

    def _format_example_comments(self, text):
        """Format example comments separately from code blocks"""
        # Find and format example usage lines
        example_pattern = r'(//\s*Example usage.*?)(?=\n```|\Z)'
        text = re.sub(example_pattern, r'</code></pre>\n<div style="margin: 10px 0; padding: 10px; background-color: #f5f7fa; border-left: 4px solid #4a6da7;">\1</div>\n<pre><code>', text, flags=re.DOTALL)
        
        # Format "How was that?" type questions
        question_pattern = r'(How was that\?.*?(?:code|explain)\.)'
        text = re.sub(question_pattern, r'<div style="margin-top: 10px; color: #555; border-top: 1px solid #eee; padding-top: 10px;">\1</div>', text)
        
        return text

    def get_system_prompt_for_model(self, model_id):
        """Return the appropriate system prompt based on model capabilities"""
        
        # Check for custom system prompt from settings
        if hasattr(self, 'custom_system_prompt_text') and self.custom_system_prompt_text:
            self.logger.info("Using custom system prompt from settings")
            return self.custom_system_prompt_text
        
        # Base system prompt for all models
        base_prompt = """You are Ladbon AI, a helpful assistant that provides accurate, direct responses. 

        CRITICAL INSTRUCTIONS:
        - NEVER invent or hallucinate fake user questions/messages
        - NEVER include User: or Assistant: prefixes in your responses
        - NEVER pretend to be both sides of a conversation
        - NEVER reference previous messages that didn't actually happen
        - ONLY respond to the actual message that was sent
        - NEVER create conversation examples unless explicitly asked
        
        When responding:
        - Answer directly without prefacing with "Ladbon AI:" or any other prefix
        - Don't ask follow-up questions unless absolutely necessary 
        - When greeting users, be warm but brief without asking how you can help
        - Format responses clearly with appropriate markdown
        - Structure code examples with proper syntax highlighting"""
        
        # Check model capabilities
        capabilities = MODEL_CAPABILITIES.get(model_id, {})
        
        # Add capability-specific instructions
        additions = []
        
        if capabilities.get("doc", False):
            additions.append("- You can analyze and discuss documents when provided")
            
        if capabilities.get("img", False):
            additions.append("- You can analyze and describe images when provided")
            
        if capabilities.get("web", False):
            additions.append("- You can use web search results to answer questions")
        
        if capabilities.get("rag", False):
            additions.append("- You can use retrieved information from documents to answer questions")
        
        # Combine base prompt with capability-specific additions
        if additions:
            return base_prompt + "\n\n" + "\n".join(additions)
        
        return base_prompt

    def _preprocess_code_blocks(self, text):
        """Ensure code blocks are properly formatted before rendering"""
        import re
        
        # Find all code blocks
        pattern = r'```([\w]*)\n(.*?)\n```'
        
        def format_code_content(match):
            lang = match.group(1).strip()
            code = match.group(2)
            
            # Detect if code has proper indentation
            lines = code.split('\n')
            if len(lines) > 3:
                indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
                if indented_lines < len(lines) * 0.5:  # Less than half the lines are indented
                    # Apply basic indentation fixes based on language
                    if lang.lower() in ['csharp', 'c#', 'cs']:
                        # Apply C# indentation rules
                        code = self._fix_csharp_indentation(code)
                    elif lang.lower() in ['python', 'py']:
                        # Apply Python indentation rules
                        code = self._fix_python_indentation(code)
            
            return f"```{lang}\n{code}\n```"
        
        # Replace each code block with properly formatted version
        return re.sub(pattern, format_code_content, text, flags=re.DOTALL)

    def _fix_csharp_indentation(self, code):
        """Apply basic C# indentation fixes"""
        lines = code.split('\n')
        result = []
        indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Decrease indent for lines with only closing braces
            if stripped == '}' or stripped == '});':
                indent = max(0, indent - 1)
            
            # Add the line with proper indentation
            result.append('    ' * indent + stripped)
            
            # Increase indent after opening braces
            if stripped.endswith('{'):
                indent += 1
                
        return '\n'.join(result)

    def save_llamacpp_settings(self):
        """Save LlamaCpp specific settings"""
        self.logger.debug("Saving LlamaCpp settings...")
        try:
            # Update n_ctx setting if available
            if hasattr(self, 'n_ctx_input'):
                new_n_ctx = self.n_ctx_input.value()
                if new_n_ctx != self.n_ctx_setting:
                    self.n_ctx_setting = new_n_ctx
                    self.logger.info(f"Updated n_ctx to {self.n_ctx_setting}")
                    
            # Update n_gpu_layers setting if available
            if hasattr(self, 'n_gpu_layers_input'):
                new_n_gpu_layers = self.n_gpu_layers_input.value()
                # Only consider this if not in CPU-only mode
                if not self.force_cpu_only and new_n_gpu_layers != self.n_gpu_layers_setting:
                    self.n_gpu_layers_setting = new_n_gpu_layers
                    self.logger.info(f"Updated n_gpu_layers to {self.n_gpu_layers_setting}")
                
            # Update force CPU mode
            if hasattr(self, 'force_cpu_checkbox'):
                new_force_cpu = self.force_cpu_checkbox.isChecked()
                if new_force_cpu != self.force_cpu_only:
                    self.force_cpu_only = new_force_cpu
                    self.logger.info(f"Updated force CPU mode to: {self.force_cpu_only}")
                    
            # Apply settings to LlamaCpp client
            if hasattr(self, 'llamacpp_client'):
                actual_n_gpu_layers = 0 if self.force_cpu_only else self.n_gpu_layers_setting
                self.llamacpp_client.update_config(n_ctx=self.n_ctx_setting, n_gpu_layers=actual_n_gpu_layers)
                self.logger.info(f"Applied settings to LlamaCpp client: n_ctx={self.n_ctx_setting}, n_gpu_layers={actual_n_gpu_layers}")
                
            # Save settings to file
            import json
            settings_path = get_settings_path()  # Use our utility function
            
            # Load existing settings if available
            try:
                if os.path.exists(settings_path):
                    with open(settings_path, "r") as f:
                        settings = json.load(f)
                else:
                    settings = {}
            except:
                settings = {}
                
            # Update with LlamaCpp settings
            settings.update({
                "n_ctx": self.n_ctx_setting,
                "n_gpu_layers": self.n_gpu_layers_setting,
                "force_cpu_only": self.force_cpu_only
            })
            
            # Write settings back to file
            with open(settings_path, "w") as f:
                json.dump(settings, f)
                
            self.show_status_message("LlamaCpp settings saved successfully", 3000)
            self.logger.info(f"LlamaCpp settings saved to file: n_ctx={self.n_ctx_setting}, n_gpu_layers={self.n_gpu_layers_setting}, force_cpu_only={self.force_cpu_only}")
            
        except Exception as e:
            self.logger.exception("Error saving LlamaCpp settings")
            self.show_status_message(f"Error saving LlamaCpp settings: {str(e)}", 3000)

    def _get_simplified_model_id(self, model_id):
        """Convert a model ID to its simplified form for better matching with LlamaCppClient's auto-detected IDs.
        For example, 'llama2-7b-chat' would convert to 'llama' to match what LlamaCppClient detects from filenames."""
        # Common model name prefixes
        prefixes = {
            "llama2": "llama",
            "mistral": "mistral",
            "phi3": "phi",
            "tinyllama": "tinyllama",
            "qwen": "qwen"
        }
        
        # Try to match the beginning of the model_id to known prefixes
        for prefix, simplified in prefixes.items():
            if model_id.lower().startswith(prefix.lower()):
                self.logger.debug(f"Simplified model ID from {model_id} to {simplified}")
                return simplified
                
        # If no known prefix is found, take the part before any dash or digit
        match = re.match(r'^([a-zA-Z]+)', model_id)
        if match:
            simplified = match.group(1).lower()
            self.logger.debug(f"Using regex to simplify model ID from {model_id} to {simplified}")
            return simplified
            
        # Fallback to the original ID
        return model_id