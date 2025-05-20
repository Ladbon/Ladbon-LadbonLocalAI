import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QLineEdit, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                             QTabWidget, QFileDialog, QLabel, QMessageBox,
                             QComboBox, QProgressBar, QStatusBar, QInputDialog, QDialog, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QTextCursor
from utils.ollama_client import OllamaClient
from cli.doc_handler import process_document
from cli.img_handler import process_image
from cli.web_search import search
from utils.logger import setup_logger
import traceback
import time  # Import time module for performance metrics

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

class LocalAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = OllamaClient()
        self.history = []
        self.context = ""
        self.context_source = ""
        
        # Set up logging
        self.logger = setup_logger('localai_gui')
        self.logger.info("LocalAI GUI application starting")
    
        # Default config
        self.current_model = "qwen3:8b"
        self.max_tokens = 8192
        self.timeout = None
        
        # Store current image path for analysis
        self.current_image_path = None
        
        self.init_ui()
        self.load_settings()
        self.setup_auto_log_refresh()  # Add this line
        self.check_ollama()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Ladbon AI Desktop - Created by Ladbon Fragari")
        self.setGeometry(100, 100, 1000, 700)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Create tab widget for the overall app (we'll keep this)
        self.tabs = QTabWidget()
    
        # Chat tab
        self.chat_tab = QWidget()
        chat_layout = QVBoxLayout()
    
        # Add a splitter for layout flexibility
        from PyQt5.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)
    
        # Chat section (left side)
        chat_section = QWidget()
        chat_section_layout = QVBoxLayout(chat_section)
    
        # Chat history
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_section_layout.addWidget(self.chat_display)
    
        # Input area
        input_layout = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
    
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(self.send_button)
        chat_section_layout.addLayout(input_layout)
    
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.hide()
        chat_section_layout.addWidget(self.progress)
    
        # Right panel (settings + logs)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
    
        # Settings section
        settings_section = QWidget()
        settings_layout = QVBoxLayout(settings_section)
    
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["qwen3:8b", "qwen3:4b", "qwen3:1.7b", "llava:7b"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
    
        # Max tokens
        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(QLabel("Max tokens:"))
        self.tokens_input = QLineEdit(str(self.max_tokens))
        tokens_layout.addWidget(self.tokens_input)
        settings_layout.addLayout(tokens_layout)
    
        # Action buttons
        settings_layout.addWidget(QLabel("Actions:"))
    
        # Web search button
        self.web_button = QPushButton("Web Search")
        self.web_button.clicked.connect(self.web_search)
        settings_layout.addWidget(self.web_button)
    
        # Document button
        self.doc_button = QPushButton("Load Document")
        self.doc_button.clicked.connect(self.load_document)
        settings_layout.addWidget(self.doc_button)
    
        # OCR Image button
        self.ocr_button = QPushButton("Extract Text from Image")
        self.ocr_button.clicked.connect(self.load_image)
        settings_layout.addWidget(self.ocr_button)
    
        # Analyze Image button
        self.analyze_img_button = QPushButton("Analyze Image (Llava)")
        self.analyze_img_button.clicked.connect(self.analyze_image)
        settings_layout.addWidget(self.analyze_img_button)
    
        # Clear history button
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self.clear_history)
        settings_layout.addWidget(self.clear_button)
    
        # Advanced settings button
        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.show_advanced_settings)
        settings_layout.addWidget(self.advanced_button)
    
        # Fast Chat button
        self.fast_chat_button = QPushButton("Switch to Fast Chat Mode")
        self.fast_chat_button.clicked.connect(self.fast_chat_mode)
        settings_layout.addWidget(self.fast_chat_button)
    
        # Add stretch to push everything up
        settings_layout.addStretch()
    
        # Add settings to right panel
        right_layout.addWidget(settings_section)
        
        # Add mini-log viewer below settings
        log_label = QLabel("Recent Logs:")
        right_layout.addWidget(log_label)
        
        self.mini_log_display = QTextEdit()
        self.mini_log_display.setReadOnly(True)
        self.mini_log_display.setMaximumHeight(200)  # Limit height
        self.mini_log_display.setLineWrapMode(QTextEdit.NoWrap)
        right_layout.addWidget(self.mini_log_display)
        
        # Add refresh button
        refresh_log_button = QPushButton("Refresh Logs")
        refresh_log_button.clicked.connect(self.refresh_mini_logs)
        right_layout.addWidget(refresh_log_button)
    
        # Add sections to splitter
        splitter.addWidget(chat_section)
        splitter.addWidget(right_panel)
    
        # Set initial sizes (70% chat, 30% settings)
        splitter.setSizes([700, 300])
    
        chat_layout.addWidget(splitter)
        self.chat_tab.setLayout(chat_layout)
        
        # Logs tab
        self.logs_tab = QWidget()
        logs_layout = QVBoxLayout()

        # Log viewer
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setLineWrapMode(QTextEdit.NoWrap)  # Better for log viewing
        logs_layout.addWidget(self.log_display)

        # Refresh button
        refresh_button = QPushButton("Refresh Logs")
        refresh_button.clicked.connect(self.refresh_logs)
        logs_layout.addWidget(refresh_button)

        self.logs_tab.setLayout(logs_layout)
        
        # Add tabs
        self.tabs.addTab(self.chat_tab, "Chat")
        self.tabs.addTab(self.logs_tab, "Logs")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Set central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    
    def check_ollama(self):
        """Check if Ollama is available"""
        if not self.client.health():
            QMessageBox.warning(self, "Ollama Not Running", 
                               "Could not connect to Ollama. Please ensure it's running.")
        else:
            self.status_bar.showMessage("Connected to Ollama")
    
    def send_message(self, message=None, with_context=False):
        """Send a message to the AI"""
        if message is None:
            message = self.message_input.text().strip()
        
        if not message:
            return
        
        # Already added if coming from web search
        if not with_context:
            self.add_to_chat("You", message)
            self.message_input.clear()
        
        # Record start time
        self.start_time = time.time()
        
        # Prepare prompt with history and context
        prompt = self._format_prompt(message, with_context)
        
        # Log the request
        self.logger.info(f"Sending message to AI using model: {self.current_model}")
        self.logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Show "thinking" indicator
        self.progress.show()
        self.send_button.setEnabled(False)
        
        # Generate response in a separate thread
        self.thread = GenerationThread(
            self.client, 
            self.current_model, 
            prompt, 
            int(self.tokens_input.text()), 
            self.timeout
        )
        self.thread.finished.connect(self._handle_response)
        self.thread.start()
    
    def _handle_response(self, response):
        """Handle the response from the AI"""
        self.progress.hide()
        self.send_button.setEnabled(True)
        
        # Calculate time elapsed and tokens/sec
        end_time = time.time()
        elapsed = end_time - self.start_time if hasattr(self, "start_time") else 0
        tokens_estimated = len(response) / 4  # rough estimate
        tokens_per_second = tokens_estimated / elapsed if elapsed > 0 else 0
        timestamp = time.strftime("%H:%M:%S")
        
        # Extract thinking process if present
        thinking = ""
        actual_response = response
        
        if "<think>" in response and "</think>" in response:
            think_start = response.find("<think>") + 7
            think_end = response.find("</think>")
            thinking = response[think_start:think_end].strip()
            actual_response = response[think_end + 8:].strip()
        
        # Handle error responses
        if response.startswith("Error:"):
            self.logger.error(f"AI response error: {response}")
            error_msg = f"<span style='color:red;'>{actual_response}</span>"
            self.chat_display.append(f"<b>AI:</b> {error_msg}")
            self.history.append({"role": "assistant", "content": actual_response})
            
            error_dialog = QMessageBox(self)
            error_dialog.setWindowTitle("Error in AI Response")
            error_dialog.setText(response)
            error_dialog.setIcon(QMessageBox.Warning)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
        else:
            # Show thinking if present
            if thinking:
                thinking_html = f"<div style='background-color:#f0f0f0;padding:8px;border-left:3px solid #ccc;color:#666;font-style:italic;'><b>AI thinking:</b><br>{thinking}</div>"
                self.chat_display.append(thinking_html)
            
            # Show actual response with performance metrics
            metrics_html = f"<div style='font-size:8pt;color:#888;'>[{timestamp} | {elapsed:.2f}s | ~{tokens_per_second:.1f} tokens/s]</div>"
            self.chat_display.append(f"<b>AI:</b> {actual_response}")
            self.chat_display.append(metrics_html)
            
            # Add to history (without thinking part)
            self.history.append({"role": "assistant", "content": actual_response})
    
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
    
    def _format_prompt(self, message, with_context=False):
        """Format the prompt with history and context"""
        prompt = ""
        
        # Add context if available and requested
        if with_context and self.context:
            prompt += f"Context:\n{self.context}\n\n"
        
        # Add history
        for msg in self.history:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
            elif msg["role"] == "system":
                # Skip system messages in the prompt
                pass
    
        # Add current message
        prompt += f"User: {message}\nAssistant: "
        return prompt
    
    def load_document(self):
        """Load a document as context"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", 
            "Documents (*.pdf *.txt *.md *.docx)"
        )
        
        if file_path:
            try:
                self.context = process_document(file_path)
                self.context_source = f"Document: {os.path.basename(file_path)}"
                self.status_bar.showMessage(f"Loaded {self.context_source}")
                QMessageBox.information(self, "Document Loaded", 
                                      f"Document loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading document: {str(e)}")
    
    def load_image(self):
        """Load an image for OCR as context"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            try:
                self.context = process_image(file_path)
                self.context_source = f"Image: {os.path.basename(file_path)}"
                self.status_bar.showMessage(f"Loaded {self.context_source}")
                QMessageBox.information(self, "Image Loaded", 
                                      f"Image text extracted: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error processing image: {str(e)}")
    
    def web_search(self):
        """Perform web search as context"""
        query, ok = QInputDialog.getText(self, "Web Search", "Enter search query:")
        
        if ok and query:
            try:
                self.progress.show()
                self.logger.info(f"Performing web search for: {query}")
                
                # Run in a thread to keep UI responsive
                class SearchThread(QThread):
                    result = pyqtSignal(str)
                    error = pyqtSignal(str)
                    
                    def __init__(self, query):
                        super().__init__()
                        self.query = query
                        
                    def run(self):
                        try:
                            from cli.web_search import search
                            results = search(self.query)
                            self.result.emit(results)
                        except Exception as e:
                            self.error.emit(f"Search error: {str(e)}")
                
                self.search_thread = SearchThread(query)
                
                def on_search_complete(results):
                    self.progress.hide()
                    self.context = results
                    self.context_source = f"Web search: {query}"
                    self.status_bar.showMessage(f"Loaded {self.context_source}")
                    
                    # Add search results to chat
                    self.add_to_chat("System", f"<i>Searched for: {query}</i>")
                    
                    # Add immediate question prompt
                    question, ok = QInputDialog.getText(
                        self, 
                        "Web Search Complete", 
                        "Search results loaded. What would you like to know about this topic?"
                    )
                    
                    if ok and question:
                        self.add_to_chat("You", question)
                        self.message_input.setText("")  # Clear input field
                        self.send_message(question, with_context=True)
                
                def on_search_error(error):
                    self.progress.hide()
                    self.logger.error(f"Web search error: {error}")
                    error_dialog = QMessageBox(self)
                    error_dialog.setWindowTitle("Search Error")
                    error_dialog.setText(error)
                    error_dialog.setIcon(QMessageBox.Critical)
                    error_dialog.setStandardButtons(QMessageBox.Ok)
                    error_dialog.exec_()
                
                self.search_thread.result.connect(on_search_complete)
                self.search_thread.error.connect(on_search_error)
                self.search_thread.start()
                
            except Exception as e:
                self.progress.hide()
                error_msg = f"Error during search: {str(e)}"
                self.logger.error(error_msg)
                QMessageBox.warning(self, "Error", error_msg)
    
    def clear_history(self):
        """Clear chat history"""
        self.history = []
        self.context = ""
        self.context_source = ""
        self.chat_display.clear()
        self.status_bar.showMessage("History cleared")
    
    def save_settings(self):
        """Save user settings to a JSON file"""
        import json
        settings = {
            "model": self.current_model,
            "max_tokens": self.max_tokens
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)

    def load_settings(self):
        """Load user settings from a JSON file"""
        import json
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
                self.current_model = settings.get("model", "qwen3:8b")
                self.max_tokens = settings.get("max_tokens", 8192)
                # Update UI
                self.model_combo.setCurrentText(self.current_model)
                self.tokens_input.setText(str(self.max_tokens))
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # Use defaults

    def analyze_image(self):
        """Analyze image content using multimodal model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if not file_path:
            return
            
        # Log the action
        self.logger.info(f"Analyzing image: {file_path}")
        
        # Show progress
        self.progress.show()
        
        try:
            # Resize image if needed to reduce memory usage
            file_path = self.resize_image_if_needed(file_path)
            self.logger.debug(f"Image prepared: {file_path}")
            self.current_image_path = file_path
            
            # Create and configure thread for image analysis
            # Store thread as instance variable to prevent garbage collection
            self.image_thread = QThread()
            
            # Create worker object
            class ImageAnalysisWorker(QObject):
                result = pyqtSignal(str)
                error = pyqtSignal(str)
                finished = pyqtSignal()
                
                def __init__(self, client, model, image_path, prompt, logger):
                    super().__init__()
                    self.client = client
                    self.model = model
                    self.image_path = image_path
                    self.prompt = prompt
                    self.logger = logger
                
                def process(self):
                    try:
                        self.logger.debug(f"Starting image analysis with model: {self.model}")
                        self.logger.debug(f"Image path: {self.image_path}")
                        self.logger.debug(f"Prompt: {self.prompt}")
                        response = self.client.generate_with_image(
                            self.model,
                            self.prompt,
                            self.image_path,
                            max_tokens=2048
                        )
                        self.logger.debug("Image analysis completed successfully")
                        self.result.emit(response)
                    except Exception as e:
                        error_trace = traceback.format_exc()
                        self.logger.error(f"Error in image analysis: {str(e)}")
                        self.logger.error(error_trace)
                        self.error.emit(f"Error analyzing image: {str(e)}\n{error_trace}")
                    finally:
                        self.finished.emit()
            
            # Create worker and move to thread
            self.image_worker = ImageAnalysisWorker(
                self.client, 
                "llava:7b", 
                file_path,
                "Please describe what you see in this image in detail.",
                self.logger
            )
            self.image_worker.moveToThread(self.image_thread)
            
            # Connect signals
            self.image_thread.started.connect(self.image_worker.process)
            self.image_worker.finished.connect(self.image_thread.quit)
            self.image_worker.finished.connect(self.image_worker.deleteLater)
            self.image_thread.finished.connect(self.image_thread.deleteLater)
            
            self.image_worker.result.connect(self.on_image_analysis_complete)
            self.image_worker.error.connect(self.on_image_analysis_error)
            
            # Start thread
            self.image_thread.start()
            
        except Exception as e:  # <- This should be aligned with the try block
            error_trace = traceback.format_exc()
            self.logger.error(f"Error in analyze_image: {str(e)}")
            self.logger.error(error_trace)
            self.progress.hide()
            error_dialog = QMessageBox(self)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText(f"Error during image analysis setup: {str(e)}")
            error_dialog.setDetailedText(error_trace)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()

    def on_image_analysis_complete(self, result):
        """Handle successful image analysis"""
        self.logger.info("Image analysis completed, updating UI")
        self.progress.hide()
        self.send_button.setEnabled(True)
        
        # Get filename for reference
        file_name = os.path.basename(self.current_image_path) if self.current_image_path else "unknown"
        
        # Extract thinking process if present
        thinking = ""
        actual_response = result
        
        if "<think>" in result and "</think>" in result:
            think_start = result.find("<think>") + 7
            think_end = result.find("</think>")
            thinking = result[think_start:think_end].strip()
            actual_response = result[think_end + 8:].strip()
        
        # Add to chat
        self.add_to_chat("You", f"[Image: {file_name}]")
        
        # Show thinking if present
        if thinking:
            thinking_html = f"<div style='background-color:#f0f0f0;padding:8px;border-left:3px solid #ccc;color:#666;font-style:italic;'><b>AI thinking:</b><br>{thinking}</div>"
            self.chat_display.append(thinking_html)
        
        # Calculate metrics
        elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        tokens_per_second = len(actual_response) / 4 / elapsed if elapsed > 0 else 0
        timestamp = time.strftime("%H:%M:%S")
        
        # Show response with metrics
        self.chat_display.append(f"<b>AI:</b> {actual_response}")
        metrics_html = f"<div style='font-size:8pt;color:#888;'>[{timestamp} | {elapsed:.2f}s | ~{tokens_per_second:.1f} tokens/s]</div>"
        self.chat_display.append(metrics_html)
        
        # Add to history
        self.history.append({"role": "user", "content": f"[Image: {file_name}]"})
        self.history.append({"role": "assistant", "content": actual_response})

    def on_image_analysis_error(self, error):
        """Handle image analysis error"""
        self.logger.error(f"Image analysis error: {error}")
        self.progress.hide()
        self.send_button.setEnabled(True)
        
        error_dialog = QMessageBox(self)
        error_dialog.setWindowTitle("Image Analysis Error")
        error_dialog.setText("Error analyzing image")
        error_dialog.setDetailedText(error)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setStandardButtons(QMessageBox.Ok)
        error_dialog.exec_()

    def resize_image_if_needed(self, image_path, max_size=512):
        """Resize large images to reduce memory usage"""
        try:
            from PIL import Image
            img = Image.open(image_path)
            if max(img.size) > max_size:
                # Calculate new dimensions while preserving aspect ratio
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                
                # Resize and save to temporary file
                img = img.resize(new_size, Image.LANCZOS)
                temp_path = os.path.join(os.path.dirname(image_path), 
                                        f"resized_{os.path.basename(image_path)}")
                img.save(temp_path)
                return temp_path
            return image_path
        except Exception:
            return image_path  # Return original if resize fails

    def on_submit(self):
        """Process user input when submit button is clicked"""
        user_input = self.input_box.toPlainText().strip()
        if not user_input:
            return
            
        # Clear input box
        self.input_box.clear()
        
        # Add user message to chat
        self.add_message("user", user_input)
        
        # Show thinking indicator
        self.response_label.setText("AI is thinking...")
        
        # Record start time for performance metrics
        start_time = time.time()
        
        try:
            # Process based on current mode
            if self.current_mode == "chat":
                response = self.client.generate(
                    model=self.current_model,
                    prompt=self._format_history_for_prompt(),
                    max_tokens=self.max_tokens
                )
            elif self.current_mode == "doc_chat":
                # Include document context
                prompt = f"Context:\n{self.current_context}\n\n" + self._format_history_for_prompt()
                response = self.client.generate(
                    model=self.current_model,
                    prompt=prompt,
                    max_tokens=self.max_tokens
                )
            elif self.current_mode == "image_chat":
                # Use image analysis
                response = self.client.generate_with_image(
                    model="llava:7b",  # Use LLaVa for images
                    prompt=user_input,
                    image_path=self.current_image_path,
                    max_tokens=2048
                )
                
            # Calculate performance metrics
            end_time = time.time()
            elapsed = end_time - start_time
            est_tokens = len(response) / 4  # rough estimate
            tokens_per_second = est_tokens / elapsed if elapsed > 0 else 0
            
            # Add timestamp and metrics
            timestamp = time.strftime("%H:%M:%S")
            performance_text = f"\n[{timestamp} | {elapsed:.2f}s | ~{tokens_per_second:.1f} tokens/s]"
            
            # Add AI response to chat with metrics
            self.add_message("assistant", response + performance_text)
            
            # Reset thinking indicator
            self.response_label.setText("")
            
            # Trim history if needed
            self._trim_history_if_needed()
            
        except Exception as e:
            self.add_message("system", f"Error: {str(e)}")
            self.response_label.setText("")

    def _trim_history_if_needed(self):
        """Trim history if it gets too long"""
        max_messages = 10
        if len(self.history) > max_messages * 2:
            self.history = self.history[-max_messages * 2:]
            self.add_to_chat("System", "[History trimmed to last 10 messages]")

    def refresh_logs(self):
        """Refresh the logs display"""
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            if not os.path.exists(logs_dir):
                self.log_display.setText("No logs directory found.")
                return
                
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            
            if not log_files:
                self.log_display.setText("No log files found.")
                return
            
            # List all log files with timestamps for selection
            log_files_with_time = [(f, os.path.getmtime(os.path.join(logs_dir, f))) for f in log_files]
            log_files_sorted = sorted(log_files_with_time, key=lambda x: x[1], reverse=True)
            
            # Get the most recent log file
            latest_log = log_files_sorted[0][0]
            log_path = os.path.join(logs_dir, latest_log)
            
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                log_content = f.read()
            
            self.log_display.setText(log_content)
            self.log_display.moveCursor(QTextCursor.End)  # Scroll to bottom
            self.status_bar.showMessage(f"Loaded log file: {latest_log}")
        except Exception as e:
            error_details = traceback.format_exc()
            self.log_display.setText(f"Error loading logs: {str(e)}\n\n{error_details}")

    def refresh_mini_logs(self):
        """Refresh the mini log display in the chat tab"""
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            if not os.path.exists(logs_dir):
                self.mini_log_display.setText("No logs directory found.")
                return
                
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            
            if not log_files:
                self.mini_log_display.setText("No log files found.")
                return
            
            # List all log files with timestamps for selection
            log_files_with_time = [(f, os.path.getmtime(os.path.join(logs_dir, f))) for f in log_files]
            log_files_sorted = sorted(log_files_with_time, key=lambda x: x[1], reverse=True)
            
            # Get the most recent log file
            latest_log = log_files_sorted[0][0]
            log_path = os.path.join(logs_dir, latest_log)
            
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                # Only read the last 20 lines for the mini display
                lines = f.readlines()
                last_lines = lines[-20:] if len(lines) > 20 else lines
                log_content = ''.join(last_lines)
            
            self.mini_log_display.setText(log_content)
            self.mini_log_display.moveCursor(QTextCursor.End)
        except Exception as e:
            self.mini_log_display.setText(f"Error loading mini logs: {str(e)}")

    def setup_auto_log_refresh(self):
        """Set up automatic log refresh timer"""
        from PyQt5.QtCore import QTimer
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self.auto_refresh_logs)
        self.log_timer.start(1000)  # Refresh every 1 second

    def auto_refresh_logs(self):
        """Automatically refresh logs if needed"""
        if self.tabs.currentWidget() == self.logs_tab:
            self.refresh_logs()
        
        # Always refresh mini logs when available
        if hasattr(self, 'mini_log_display') and self.mini_log_display.isVisible():
            self.refresh_mini_logs()

    def _on_model_changed(self, model_name):
        """Handle model selection change"""
        self.current_model = model_name
        self.logger.info(f"Changed model to: {model_name}")
        self.status_bar.showMessage(f"Model changed to: {model_name}")

    def show_advanced_settings(self):
        """Show advanced settings dialog"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QComboBox, QSlider
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Settings")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Temperature setting with slider for more precise control
        temp_layout = QVBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        
        temp_value_label = QLabel("0.3")
        temp_slider = QSlider(Qt.Horizontal)
        temp_slider.setMinimum(1)
        temp_slider.setMaximum(10)
        temp_slider.setValue(3)  # Default 0.3
        temp_slider.setTickPosition(QSlider.TicksBelow)
        
        def update_temp_label(value):
            temp_value = value / 10.0
            temp_value_label.setText(f"{temp_value:.1f}")
        
        temp_slider.valueChanged.connect(update_temp_label)
        
        temp_labels_layout = QHBoxLayout()
        temp_labels_layout.addWidget(QLabel("More focused"))
        temp_labels_layout.addStretch()
        temp_labels_layout.addWidget(QLabel("More creative"))
        
        temp_slider_layout = QHBoxLayout()
        temp_slider_layout.addWidget(temp_slider)
        temp_slider_layout.addWidget(temp_value_label)
        
        temp_layout.addLayout(temp_labels_layout)
        temp_layout.addLayout(temp_slider_layout)
        layout.addLayout(temp_layout)
    
        # Unload models option
        unload_button = QPushButton("Unload Models to Save Memory")
        unload_button.clicked.connect(self._show_unload_models_dialog)
        layout.addWidget(unload_button)
        
        # System prompt customization
        sys_layout = QVBoxLayout()
        sys_layout.addWidget(QLabel("Custom System Prompt:"))
        sys_prompt = QTextEdit()
        sys_prompt.setPlaceholderText("You are Ladbon AI, an AI assistant created by Ladbon Fragari.")
        sys_prompt.setMaximumHeight(100)
        sys_layout.addWidget(sys_prompt)
        layout.addLayout(sys_layout)
        
        # Save and close buttons
        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        close_button = QPushButton("Close")
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(close_button)
        layout.addLayout(buttons_layout)
        
        # Apply settings on save
        def on_save():
            # Save temperature
            temp = temp_slider.value() / 10.0
            self.logger.info(f"Temperature set to {temp}")
            # Apply settings
            dialog.accept()
        
        # Connect buttons
        close_button.clicked.connect(dialog.reject)
        save_button.clicked.connect(on_save)
        
        # Show the dialog
        dialog.exec_()

    def _show_unload_models_dialog(self):
        """Show dialog to unload models"""
        import subprocess
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Unload Models")
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select models to unload:"))
        
        # Get available models
        try:
            models_output = subprocess.check_output(["ollama", "list"], text=True)
            model_lines = models_output.strip().split('\n')[1:]  # Skip header
            models = []
            
            for line in model_lines:
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
        except Exception as e:
            models = ["qwen3:8b", "qwen3:4b", "qwen3:1.7b", "llava:7b"]
            self.logger.error(f"Error getting model list: {str(e)}")
        
        # Create checkboxes for each model
        checkboxes = []
        for model in models:
            checkbox = QCheckBox(model)
            checkboxes.append((checkbox, model))
            layout.addWidget(checkbox)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        unload_button = QPushButton("Unload Selected")
        cancel_button = QPushButton("Cancel")
        buttons_layout.addWidget(unload_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
        # Connect buttons
        cancel_button.clicked.connect(dialog.reject)
        
        def on_unload():
            for checkbox, model in checkboxes:
                if checkbox.isChecked():
                    try:
                        self.logger.info(f"Unloading model: {model}")
                        subprocess.run(["ollama", "rm", model])
                        self.status_bar.showMessage(f"Unloaded model: {model}")
                    except Exception as e:
                        self.logger.error(f"Error unloading model {model}: {str(e)}")
            dialog.accept()
        
        unload_button.clicked.connect(on_unload)
        
        # Show the dialog
        dialog.exec_()

    # Add this method to allow fast chat in GUI, similar to CLI

    def fast_chat_mode(self):
        """Switch to fast chat mode using smaller models"""
        # Show confirmation dialog
        msg = QMessageBox()
        msg.setWindowTitle("Switch to Fast Chat Mode")
        msg.setText("This will switch to using qwen3:1.7b for faster responses.")
        msg.setInformativeText("Current history will be preserved but responses may be shorter and less detailed.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        
        if msg.exec_() == QMessageBox.Ok:
            # Save current model
            self.previous_model = self.current_model
            
            # Switch to fast model
            self.current_model = "qwen3:1.7b"
            self.model_combo.setCurrentText(self.current_model)
            
            # Update tokens for faster responses
            self.tokens_input.setText("1024")
            
            # Update status
            self.status_bar.showMessage("Fast Chat Mode Activated - Using qwen3:1.7b")
            self.add_to_chat("System", "<i>Switched to Fast Chat mode (qwen3:1.7b)</i>")

def main():
    app = QApplication(sys.argv)
    window = LocalAIApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()