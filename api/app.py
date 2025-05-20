import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QTabWidget, QFileDialog, QLabel, QMessageBox,
                             QComboBox, QProgressBar, QStatusBar, QCheckBox,
                             QListWidget, QListWidgetItem, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QTextCursor
from utils.ollama_client import OllamaClient
from cli.doc_handler import process_document
from cli.img_handler import process_image, query_image
from cli.web_search import search
from utils.logger import setup_logger
import traceback
import time

# Define model capabilities - will be used to enable/disable features
MODEL_CAPABILITIES = {
    # Llama models
    "llama3:8b": {
        "description": "Llama 3 8B (Chat, Docs, Web)",
        "doc": True,
        "img": False,
        "ocr": False,
        "web": True,
        "rag": True
    },
    "llama3:70b": {
        "description": "Llama 3 70B (Chat, Docs, Web, RAG)",
        "doc": True,
        "img": False,
        "ocr": False,
        "web": True,
        "rag": True
    },
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
    
    def __init__(self, client, model, prompt, max_tokens, timeout):
        super().__init__()
        self.client = client
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout = timeout
        
    def run(self):
        try:
            response = self.client.generate(
                model=self.model,
                prompt=self.prompt,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class ImageAnalysisThread(QThread):
    """Thread for analyzing images"""
    finished = pyqtSignal(str)
    
    def __init__(self, client, model, prompt, image_path, max_tokens=2048):
        super().__init__()
        self.client = client
        self.model = model
        self.prompt = prompt
        self.image_path = image_path
        self.max_tokens = max_tokens
        
    def run(self):
        try:
            response = self.client.generate_with_image(
                model=self.model,
                prompt=self.prompt,
                image_path=self.image_path,
                max_tokens=self.max_tokens
            )
            self.finished.emit(response)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")

class LocalAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = OllamaClient()
        self.history = []
        self.selected_docs = []
        self.selected_imgs = []
        self.web_search_enabled = False  # Track if web search is enabled for the next message
        
        # Set up logging
        self.logger = setup_logger('Ladbon AI GUI')
        self.logger.info("Ladbon AI GUI application starting")

        # Default config
        self.current_model = "qwen3:8b"
        self.max_tokens = 8192
        self.timeout = None
        
        self.init_ui()
        self.load_settings()
        # Add this line to update the UI based on loaded model capabilities
        self.update_ui_for_model_capabilities()  
        self.check_ollama()
        
        # Set up log auto-refresh timer
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.load_logs)
        self.log_timer.start(1000)  # Refresh every second
    
    def init_ui(self):
        """Initialize the user interface with document and image lists"""
        self.setWindowTitle("Ladbon AI Desktop - Created by Ladbon Fragari")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create a splitter for the main area
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Chat area
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
        self.web_button.setCheckable(True)  # Make it a toggle button
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
        
        # Right side: Model selection and document/image lists
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
        
        # Clear selections button
        clear_button = QPushButton("Clear Selections")
        clear_button.clicked.connect(self.clear_selections)
        right_layout.addWidget(clear_button)
        
        # Clear history button
        clear_history_button = QPushButton("Clear Chat History")
        clear_history_button.clicked.connect(self.clear_history)
        right_layout.addWidget(clear_history_button)
        
        # Add the widgets to the splitter
        self.main_splitter.addWidget(chat_widget)
        self.main_splitter.addWidget(right_widget)
        
        # Set the splitter sizes (70% chat, 30% sidebar)
        self.main_splitter.setSizes([700, 300])
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)
        
        # Tabs for settings
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add a Settings tab
        self.settings_tab = QWidget()
        self.tabs.addTab(self.settings_tab, "Settings")
        settings_layout = QVBoxLayout(self.settings_tab)
        
        # Move token settings to this tab
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max tokens:"))
        self.tokens_input = QLineEdit(str(int(self.max_tokens)))  # Force integer conversion
        tokens_layout.addWidget(self.tokens_input)
        settings_layout.addLayout(tokens_layout)
        
        # Add save button
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        settings_layout.addWidget(save_settings_btn)
        settings_layout.addStretch()
        
        # Add Logs tab
        self.logs_tab = QWidget()
        self.tabs.addTab(self.logs_tab, "Logs")
        logs_layout = QVBoxLayout(self.logs_tab)

        self.logs_display = QTextEdit()
        self.logs_display.setReadOnly(True)
        logs_layout.addWidget(self.logs_display)

        # Add a refresh button
        refresh_logs_btn = QPushButton("Refresh Logs")
        refresh_logs_btn.clicked.connect(self.load_logs)
        logs_layout.addWidget(refresh_logs_btn)

        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Load documents and images
        self.load_documents()
        self.load_images()
    
    def update_model_list(self):
        """Get installed models from Ollama and add them to the combo box with descriptions"""
        try:
            # Get installed models
            installed_models = self.client.list_models()
            
            if not installed_models:
                self.logger.warning("No models found in Ollama. Please install models first.")
                try:
                    self.statusBar().showMessage("No models found in Ollama")
                except:
                    pass  # Skip if status bar isn't ready yet
                return
                
            self.logger.info(f"Found {len(installed_models)} installed models: {installed_models}")
            
            # Clear the combo box
            self.model_combo.clear()
            
            # Add only models that are installed
            for model in installed_models:
                # Some models might have different naming in Ollama vs our capability list
                # Try to find a match or add with generic capabilities
                if model in MODEL_CAPABILITIES:
                    desc = MODEL_CAPABILITIES[model]["description"]
                    self.model_combo.addItem(f"{model} - {desc}", model)
                else:
                    # For models not in our capability list, add with a generic description
                    self.model_combo.addItem(f"{model} - (Basic Chat)", model)
        
        except Exception as e:
            self.logger.error(f"Error loading model list: {str(e)}")
            try:
                self.statusBar().showMessage("Error loading models from Ollama")
            except:
                pass  # Skip if status bar isn't ready yet
    
    def load_documents(self):
        """Load documents from docs folder and add them to the list with checkboxes"""
        self.doc_list.clear()
        
        # Fix path to docs folder (inside src folder)
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        
        # Create docs directory if it doesn't exist
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
            self.logger.info(f"Created docs directory: {docs_dir}")
        
        # Check if directory exists and is accessible
        if os.path.isdir(docs_dir):
            try:
                # Get list of documents - CASE INSENSITIVE CHECK
                docs = [f for f in os.listdir(docs_dir) if f.lower().endswith(('.pdf', '.txt', '.md', '.docx', '.doc'))]
                
                # Add each document to the list with a checkbox
                for doc in docs:
                    item = QListWidgetItem(doc)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.doc_list.addItem(item)
                
                self.doc_list.itemChanged.connect(self.on_doc_selection_changed)
                self.logger.info(f"Loaded {len(docs)} documents from {docs_dir}")
            
            except Exception as e:
                self.logger.error(f"Error loading documents: {str(e)}")
                self.statusBar().showMessage(f"Error loading documents: {str(e)}")
        else:
            self.logger.warning(f"Documents directory not found: {docs_dir}")
    
    def load_images(self):
        """Load images from img folder and add them to the list with checkboxes"""
        self.img_list.clear()
        
        # Fix path to img folder (inside src folder)
        img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img")
        
        # Create img directory if it doesn't exist
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            self.logger.info(f"Created img directory: {img_dir}")
        
        # Check if directory exists and is accessible
        if os.path.isdir(img_dir):
            try:
                # Get list of images
                imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                # Add each image to the list with a checkbox
                for img in imgs:
                    item = QListWidgetItem(img)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.img_list.addItem(item)
                
                self.img_list.itemChanged.connect(self.on_img_selection_changed)
                self.logger.info(f"Loaded {len(imgs)} images from {img_dir}")
            
            except Exception as e:
                self.logger.error(f"Error loading images: {str(e)}")
                self.statusBar().showMessage(f"Error loading images: {str(e)}")
        else:
            self.logger.warning(f"Images directory not found: {img_dir}")
    
    def on_doc_selection_changed(self, item):
        """Handle document selection change"""
        doc_name = item.text()
        # Fix path to match where we're loading from
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        doc_path = os.path.join(docs_dir, doc_name)
        
        if item.checkState() == Qt.Checked:
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
        # Fix path to match where we're loading from
        img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img")
        img_path = os.path.join(img_dir, img_name)
        
        if item.checkState() == Qt.Checked:
            if img_path not in self.selected_imgs:
                self.selected_imgs.append(img_path)
                self.logger.debug(f"Selected image: {img_name}")
        else:
            if img_path in self.selected_imgs:
                self.selected_imgs.remove(img_path)
                self.logger.debug(f"Unselected image: {img_name}")
    
    def on_model_changed(self, model_text):
        """Handle model selection change and update UI accordingly"""
        # Extract model name from combo box text (which includes description)
        model_name = self.model_combo.currentData()
        if not model_name:
            # If data() doesn't work, parse it from text
            if " - " in model_text:
                model_name = model_text.split(" - ")[0]
            else:
                model_name = model_text
        
        self.current_model = model_name
        self.logger.info(f"Changed model to: {model_name}")
        
        # Update UI based on model capabilities
        self.update_ui_for_model_capabilities()
        
        self.statusBar().showMessage(f"Model changed to: {model_name}")
    
    def update_ui_for_model_capabilities(self):
        """Update UI elements based on current model capabilities"""
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
            
            # If model can't use docs, uncheck and disable
            if not can_use_docs:
                item.setCheckState(Qt.Unchecked)
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            else:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
        
        # Enable/disable image list items
        can_use_imgs = capabilities.get("img", False) or capabilities.get("ocr", False)
        for i in range(self.img_list.count()):
            item = self.img_list.item(i)
            
            # If model can't use images, uncheck and disable
            if not can_use_imgs:
                item.setCheckState(Qt.Unchecked)
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            else:
                item.setFlags(item.flags() | Qt.ItemIsEnabled)
    
    def check_ollama(self):
        """Check if Ollama is available"""
        if not self.client.health():
            QMessageBox.warning(self, "Ollama Not Running", 
                               "Could not connect to Ollama. Please ensure it's running.")
            self.statusBar().showMessage("Error: Ollama not running")
        else:
            self.statusBar().showMessage("Connected to Ollama")
    
    def send_message(self):
        """Send a message to the AI with selected documents and images"""
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Clear input field
        self.message_input.clear()
        
        # Add user message to chat
        self.add_to_chat("You", message)
        
        # Record start time
        self.start_time = time.time()
        
        # Show progress bar
        self.progress.show()
        self.send_button.setEnabled(False)
        
        # Process selected documents and images
        context = self.build_context_from_selections()
        
        # Add web search results if enabled
        if self.web_search_enabled:
            try:
                self.statusBar().showMessage("Searching the web...")
                web_results = search(message, max_results=5)
                if web_results and len(web_results) > 10:  # Only use meaningful results
                    context = f"{context}\n\n[Web Search Results]:\n{web_results}" if context else f"[Web Search Results]:\n{web_results}"
                    self.chat_display.append(f"<span style='color: #4CAF50;'><i>🔍 Web search performed</i></span>")
                else:
                    self.chat_display.append(f"<span style='color: orange;'><i>🔍 No useful web results found</i></span>")
            except Exception as e:
                self.logger.error(f"Web search error: {str(e)}")
                self.chat_display.append(f"<span style='color: red;'><i>🔍 Web search failed: {str(e)}</i></span>")
            
            # Reset web search toggle for next message
            self.web_search_enabled = False
            self.web_button.setChecked(False)
            self.web_button.setStyleSheet("")
        
        # Build the final prompt
        if context:
            prompt = f"Context:\n{context}\n\nUser: {message}"
        else:
            prompt = self._format_chat_history() + f"User: {message}"
        
        # Check if we have selected images for visual analysis
        if self.selected_imgs and MODEL_CAPABILITIES.get(self.current_model, {}).get("img", False):
            # Use the first selected image for visual analysis
            img_path = self.selected_imgs[0]
            
            # Start image analysis thread
            self.img_thread = ImageAnalysisThread(
                self.client,
                self.current_model if "llama4" in self.current_model or "phi3" in self.current_model else "llava:7b",
                message,
                img_path,
                max_tokens=int(self.tokens_input.text())
            )
            self.img_thread.finished.connect(self._handle_response)
            self.img_thread.start()
        else:
            # Get max_tokens from the input field
            try:
                # Clean the input by removing all non-digit characters
                max_tokens_text = ''.join(c for c in self.tokens_input.text() if c.isdigit())
                max_tokens = int(max_tokens_text) if max_tokens_text else 8192
                if max_tokens <= 0:
                    max_tokens = 8192  # Default if invalid
            except Exception as e:
                self.logger.warning(f"Could not parse max_tokens: {str(e)}, using default")
                max_tokens = 8192  # Default if parsing fails
            
            self.logger.info(f"Using max_tokens: {max_tokens}")
            
            # Start regular text generation thread
            self.thread = GenerationThread(
                self.client,
                self.current_model,
                prompt,
                max_tokens,  # Use the max_tokens we just validated
                self.timeout
            )
            self.thread.finished.connect(self._handle_response)
            self.thread.start()
    
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
    
    def _handle_response(self, response):
        """Handle the response from the AI"""
        self.progress.hide()
        self.send_button.setEnabled(True)
        
        # Calculate time elapsed
        elapsed = time.time() - self.start_time if hasattr(self, "start_time") else 0
        
        # Add response to chat
        self.add_to_chat("AI", response)
        
        # Add performance metrics
        tokens_estimated = len(response) / 4  # Rough estimate
        tokens_per_second = tokens_estimated / elapsed if elapsed > 0 else 0
        metrics = f"[Generated in {elapsed:.2f}s, ~{tokens_per_second:.1f} tokens/s]"
        self.chat_display.append(f"<span style='color: gray; font-size: 8pt;'>{metrics}</span>")
        
        # Add to history
        self.history.append({"role": "assistant", "content": response})
        
        # Trim history if needed
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def add_to_chat(self, sender, message):
        """Add a message to the chat display"""
        if sender == "You":
            # Add to history
            self.history.append({"role": "user", "content": message})
            self.chat_display.append(f"<b>{sender}:</b> {message}")
        else:
            self.chat_display.append(f"<b>{sender}:</b> {message}")
        
        # Scroll to bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
    
    def clear_selections(self):
        """Clear all document and image selections"""
        # Uncheck all documents
        for i in range(self.doc_list.count()):
            self.doc_list.item(i).setCheckState(Qt.Unchecked)
        
        # Uncheck all images
        for i in range(self.img_list.count()):
            self.img_list.item(i).setCheckState(Qt.Unchecked)
        
        self.selected_docs = []
        self.selected_imgs = []
        self.statusBar().showMessage("Selections cleared")
    
    def clear_history(self):
        """Clear chat history"""
        self.history = []
        self.chat_display.clear()
        self.statusBar().showMessage("Chat history cleared")
    
    def load_settings(self):
        """Load user settings from a JSON file"""
        import json
        try:
            settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
            if os.path.exists(settings_path):
                with open(settings_path, "r") as f:
                    settings = json.load(f)
                    self.current_model = settings.get("model", "qwen3:8b")
                    self.max_tokens = settings.get("max_tokens", 8192)
                    # Update UI
                    self.tokens_input.setText(str(self.max_tokens))
                    
                    # Find and select the current model in combo box
                    for i in range(self.model_combo.count()):
                        if self.current_model in self.model_combo.itemText(i):
                            self.model_combo.setCurrentIndex(i)
                            break
        except Exception as e:
            self.logger.error(f"Error loading settings: {str(e)}")
    
    def save_settings(self):
        """Save user settings to a JSON file"""
        import json
        try:
            # Update the max tokens value from input - clean it first
            try:
                max_tokens_text = ''.join(c for c in self.tokens_input.text() if c.isdigit())
                new_max_tokens = int(max_tokens_text) if max_tokens_text else self.max_tokens
                if new_max_tokens > 0:
                    self.max_tokens = new_max_tokens
                    self.logger.info(f"Updated max tokens to {self.max_tokens}")
                else:
                    self.logger.warning("Invalid max tokens value: must be > 0")
            except ValueError as e:
                self.logger.warning(f"Invalid max tokens value: {str(e)}")
            
            # Save settings to file
            settings = {
                "model": self.current_model,
                "max_tokens": self.max_tokens
            }
            settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.json")
            with open(settings_path, "w") as f:
                json.dump(settings, f)
                
            self.statusBar().showMessage("Settings saved successfully", 3000)
            self.logger.info(f"Settings saved: model={self.current_model}, max_tokens={self.max_tokens}")
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}")
            self.statusBar().showMessage(f"Error saving settings: {str(e)}", 3000)
    
    def load_logs(self):
        """Load the most recent log file with newest entries at the top"""
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            if not os.path.exists(logs_dir):
                self.logs_display.setPlainText("No logs directory found.")
                return
                
            log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) 
                         if f.startswith('localai_') and f.endswith('.log')]
            
            if not log_files:
                self.logs_display.setPlainText("No log files found.")
                return
                
            recent_log = max(log_files, key=os.path.getmtime)
            
            with open(recent_log, 'r') as f:
                # Read all lines and reverse them (newest first)
                log_lines = f.readlines()
                log_lines.reverse()
                log_content = ''.join(log_lines)
                self.logs_display.setPlainText(log_content)
                
            self.statusBar().showMessage(f"Loaded logs from {os.path.basename(recent_log)}")
        except Exception as e:
            self.logs_display.setPlainText(f"Error loading logs: {str(e)}")
    
    def closeEvent(self, event):
        """Save settings when the application is closed"""
        self.save_settings()
        super().closeEvent(event)

    def toggle_web_search(self):
        """Toggle web search for the next message"""
        self.web_search_enabled = not self.web_search_enabled
        
        if self.web_search_enabled:
            self.web_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.statusBar().showMessage("Web search enabled for next message")
            self.logger.info("Web search enabled")
        else:
            self.web_button.setStyleSheet("")
            self.statusBar().showMessage("Web search disabled")
            self.logger.info("Web search disabled")
def main():
    app = QApplication(sys.argv)
    window = LocalAIApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()