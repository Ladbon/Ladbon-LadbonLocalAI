import sys, os, time
from typing import List, Dict
from utils.ollama_client import OllamaClient

class ChatSession:
    def __init__(self, client: OllamaClient, model: str = "qwen3:8b", 
         max_tokens: int = 10000, timeout: int = None):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.history: List[Dict[str, str]] = []
        
        # Check Ollama availability
        if not client.health():
            print("⚠️ Ollama service not detected. Attempting to start...")
            self._ensure_ollama()
            if not client.health():
                print("❌ Failed to start Ollama. Please start it manually and retry.")
                sys.exit(1)
        print(f"✓ Connected to Ollama with model: {model}")
    
    def _ensure_ollama(self):
        """Attempt to start the Ollama service if it's not running"""
        import subprocess
        import time
        
        # Kill any existing processes
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Start Ollama
        subprocess.Popen(["ollama", "serve"], shell=True)
        time.sleep(5)  # Wait for startup
    
    def menu(self) -> str:
        """Display menu and get user choice"""
        print("\n" + "="*50)
        print("🤖 Ladbon AI - Created by Ladbon Fragari")
        print("="*50)
        print("1. Chat Mode (Qwen3)")
        print("2. Document Chat (Load & Discuss PDF/TXT)")
        print("3. Image OCR Chat (Extract & Discuss Text)")
        print("4. Image Analysis Chat (Describe & Discuss)")
        print("5. Web Search Chat (Research & Discuss)")
        print("6. Fast Chat (Qwen3 1.7B)")
        print("7. Clear History")
        print("8. Exit")
        print("9. Advanced Options")
        
        while True:
            choice = input("\nEnter your choice (1-9): ")
            if choice == "1":
                return "Chat"
            elif choice == "2":
                return "Document"
            elif choice == "3":
                return "ImageOCR"
            elif choice == "4":
                return "ImageAnalysis"  # New option
            elif choice == "5":
                return "WebSearch"
            elif choice == "6":
                return "FastChat"
            elif choice == "7":
                return "ClearHistory"
            elif choice == "8":
                return "Exit"
            elif choice == "9":
                return "AdvancedOptions"
            else:
                print("Invalid choice. Please try again.")
    
    def _list_and_select_files(self, directory, extensions=None):
        """List files with specific extensions from a directory and let user select one"""
        try:
            # Ensure directory exists
            if not os.path.exists(directory):
                print(f"Directory not found: {directory}")
                return None
                
            # List files with specified extensions
            files = []
            for file in os.listdir(directory):
                if extensions is None or any(file.lower().endswith(ext.lower()) for ext in extensions):
                    files.append(file)
                    
            if not files:
                print(f"No suitable files found in {directory}")
                return None
                
            # Display files with numbers
            print(f"\nAvailable files in {os.path.basename(directory)}:")
            for i, file in enumerate(files, 1):
                print(f"{i}. {file}")
                
            # Get user selection
            while True:
                try:
                    choice = input("\nSelect a file number (or 0 to cancel): ")
                    if choice == '0':
                        return None
                        
                    choice = int(choice)
                    if 1 <= choice <= len(files):
                        return os.path.join(directory, files[choice-1])
                    else:
                        print(f"Please select a number between 1 and {len(files)}")
                except ValueError:
                    print("Please enter a valid number")
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            return None

    def handle(self, choice: str):
        """Handle the selected menu option"""
        if choice == "Chat":
            self._chat_mode()
        elif choice == "Document":
            from cli.doc_handler import process_document
            
            # Get docs directory
            docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
            doc_path = self._list_and_select_files(docs_dir, [".pdf", ".txt", ".md", ".docx", ".doc"])
            
            if doc_path:
                context = process_document(doc_path)
                self._chat_with_context(context, f"Document: {os.path.basename(doc_path)}")
            else:
                print("Document selection canceled")
        elif choice == "ImageOCR":
            from cli.img_handler import process_image
            
            # Get images directory
            img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img")
            img_path = self._list_and_select_files(img_dir, [".jpg", ".jpeg", ".png", ".bmp", ".gif"])
            
            if img_path:
                context = process_image(img_path)
                self._chat_with_context(context, f"Image OCR: {os.path.basename(img_path)}")
            else:
                print("Image selection canceled")
        elif choice == "ImageAnalysis":
            from cli.img_handler import query_image
    
            # Get images directory
            img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img")
            img_path = self._list_and_select_files(img_dir, [".jpg", ".jpeg", ".png", ".bmp", ".gif"])
    
            if img_path:
                # Use the continuous image chat mode
                self._chat_with_image_analysis(img_path)
            else:
                print("Image selection canceled")
        elif choice == "WebSearch":
            from cli.web_search import search
    
            print("\nYou can ask questions that require up-to-date information.")
            print("The AI will search the web automatically to answer your question.")
            print("Type 'exit' to return to the main menu.")
    
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() == 'exit':
                    break
                    
                if not user_input.strip():
                    continue
                
                self._add_message("user", user_input)
                
                print("\nAI is searching the web...")
                
                # Generate search terms based on the user's question
                search_query = self._generate_search_query(user_input)
                print(f"Searching for: {search_query}")
                
                # Perform the search
                context = search(search_query)
                
                # Process the response with the search results as context
                prompt = f"Context from web search for '{search_query}':\n{context}\n\n" + self._format_history()
                
                print("\nAI is thinking...")
                start_time = time.time()
                self._trim_history_if_needed()

                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                self._add_message("assistant", response)
                
                # Calculate performance metrics
                est_prompt_tokens = len(prompt) / 4
                est_response_tokens = len(response) / 4
                total_tokens = est_prompt_tokens + est_response_tokens
                tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
                
                # Print response with performance metrics
                timestamp = time.strftime("%H:%M:%S")
                print(f"\nAI: {response}")
                print(f"\n[{timestamp} | {elapsed:.2f}s | ~{tokens_per_second:.1f} tokens/s]")
        elif choice == "FastChat":
            # Call the fast chat mode with Mistral
            self._fast_chat_mode()
        elif choice == "ClearHistory":
            self.history = []
            print("Chat history cleared.")
        elif choice == "AdvancedOptions":
            self._show_advanced_options()
    
    def _chat_mode(self):
        """Simple chat mode with continuous conversation"""
        print("\nYou are now in chat mode. Type 'exit' to return to the main menu.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Returning to main menu...")
                break
                
            if not user_input.strip():
                continue
            
            self._add_message("user", user_input)
            prompt = self._format_history()
            
            print("\nAI is thinking...")
            start_time = time.time()
            self._trim_history_if_needed()

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            self._add_message("assistant", response)
            
            # Calculate approximate tokens per second
            est_prompt_tokens = len(prompt) / 4  # rough estimate
            est_response_tokens = len(response) / 4  # rough estimate
            total_tokens = est_prompt_tokens + est_response_tokens
            tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
            
            # Print response with performance metrics
            timestamp = time.strftime("%H:%M:%S")
            print(f"\nAI: {response}")
            print(f"\n[{timestamp} | {elapsed:.2f}s | ~{tokens_per_second:.1f} tokens/s]")
    
    def _chat_with_context(self, context: str, source: str):
        """Chat with additional context from document, image, or web search"""
        print(f"\nUsing context from: {source}")
        print("You are now in context chat mode. Type 'exit' to return to the main menu.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Returning to main menu...")
                break
                
            if not user_input.strip():
                continue
            
            self._add_message("user", user_input)
            
            # Include the context in the prompt
            prompt = f"Context:\n{context}\n\n" + self._format_history()
            
            print("\nAI is thinking...")
            start_time = time.time()
            self._trim_history_if_needed()

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            self._add_message("assistant", response)
            
            # Calculate approximate tokens per second
            est_prompt_tokens = len(prompt) / 4  # rough estimate
            est_response_tokens = len(response) / 4  # rough estimate
            total_tokens = est_prompt_tokens + est_response_tokens
            tokens_per_second = total_tokens / elapsed if elapsed > 0 else 0
            
            # Print response with performance metrics
            timestamp = time.strftime("%H:%M:%S")
            print(f"\nAI: {response}")
            print(f"\n[{timestamp} | {elapsed:.2f}s | ~{tokens_per_second:.1f} tokens/s]")
    
    def _add_message(self, role: str, content: str):
        """Add a message to the chat history"""
        self.history.append({"role": role, "content": content})
    
    def _format_history(self) -> str:
        """Format chat history into a prompt for the model"""
        formatted = ""
        for msg in self.history:
            if msg["role"] == "user":
                formatted += f"User: {msg['content']}\n"
            else:
                formatted += f"Assistant: {msg['content']}\n"
        
        formatted += "Assistant: "
        return formatted
    
    def _chat_with_image_analysis(self, img_path: str):
        """Chat using an image with visual understanding capabilities"""
        print(f"\nUsing visual analysis on: {os.path.basename(img_path)}")
        print("You can chat with the AI about this image. Type 'exit' to return to menu.")
        
        # First, let the AI describe the image on its own
        print("\nAI is analyzing the image...")
        from cli.img_handler import query_image
        initial_response = query_image(self.client, img_path, "Describe this image in detail.", "llava:7b")
        print(f"\nAI: {initial_response}")
        
        # Add to history
        self._add_message("user", f"[Image: {os.path.basename(img_path)}] Describe this image.")
        self._add_message("assistant", initial_response)
        
        # Continue conversation about the image
        while True:
            query = input("\nYou (type 'exit' to return to menu): ")
            if query.lower() == 'exit':
                break
                
            print("\nAI is thinking...")
            
            response = query_image(self.client, img_path, query, "llava:7b")
            
            print(f"\nAI: {response}")
            
            # Add to history
            self._add_message("user", f"[Image: {os.path.basename(img_path)}] {query}")
            self._add_message("assistant", response)
    
    def _show_advanced_options(self):
        """Show advanced options menu"""
        print("\nAdvanced Options:")
        print("1. Unload models to save memory")
        print("2. Change current model")
        print("3. Back to main menu")
        
        while True:
            choice = input("\nEnter your choice (1-3): ")
            if choice == "1":
                import subprocess
                models = ["llava:7b", "optimized-llava:latest", "qwen3:8b", "qwen3:4b", "qwen3:1.7b"]
                
                print("Available models to unload:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                print(f"{len(models)+1}. All models")
                
                try:
                    model_choice = int(input("\nSelect a model to unload (or 0 to cancel): "))
                    if model_choice == 0:
                        break
                    elif 1 <= model_choice <= len(models):
                        subprocess.run(["ollama", "rm", models[model_choice-1]])
                        print(f"Unloaded {models[model_choice-1]}")
                    elif model_choice == len(models)+1:
                        for model in models:
                            subprocess.run(["ollama", "rm", model])
                        print("All models unloaded")
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Please enter a valid number")
                break
            elif choice == "2":
                available_models = ["qwen3:8b", "qwen3:4b", "qwen3:1.7b", "llava:7b"]
                print("\nAvailable models:")
                for i, m in enumerate(available_models, 1):
                    print(f"{i}. {m}")
                
                try:
                    model_choice = int(input("\nSelect a model (or 0 to cancel): "))
                    if 1 <= model_choice <= len(available_models):
                        self.model = available_models[model_choice-1]
                        print(f"Model changed to: {self.model}")
                except ValueError:
                    print("Invalid input")
                break
            elif choice == "3":
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _generate_search_query(self, user_question: str) -> str:
        """Generate a search query from the user's question"""
        # For simple questions, we can just use them directly
        if len(user_question) < 60 and ("?" in user_question or user_question.lower().startswith("who") or 
                                        user_question.lower().startswith("what") or 
                                        user_question.lower().startswith("when") or
                                        user_question.lower().startswith("where") or
                                        user_question.lower().startswith("how")):
            return user_question
        
        # For more complex questions, ask the AI to extract search terms
        prompt = f"""Extract the most important search terms from this question. 
Return ONLY the search terms, nothing else.
Question: {user_question}
Search terms:"""
        
        self._trim_history_if_needed()
        
        search_terms = self.client.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=100,
            timeout=self.timeout
        )
        
        return search_terms.strip()
    
    def _trim_history_if_needed(self):
        """Trim history if it gets too long (to avoid context length issues)"""
        max_messages = 10  # Keep last 10 message pairs
        if len(self.history) > max_messages * 2:
            # Keep the last X messages
            self.history = self.history[-max_messages * 2:]
            self._add_message("system", "[History trimmed to last 10 messages]")
    
    def _fast_chat_mode(self):
        """Fast chat mode using qwen3:1.7b"""
        print("\nYou are now in Fast Chat mode with qwen3:1.7b. Type 'exit' to return to the main menu.")
        
        # Save original model and use the small model
        original_model = self.model
        self.model = "qwen3:1.7b"
        
        # Use more aggressive settings for fast responses
        payload_options = {
            "num_gpu": 1,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 20,
            "num_thread": 6,
            "num_ctx": 1024,  # Smaller context
            "repeat_penalty": 1.1
        }
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Returning to main menu...")
                break
                
            if not user_input.strip():
                continue
            
            # Use a minimal prompt for faster responses
            prompt = f"User: {user_input}\nAssistant:"
            
            print("\nAI is thinking...")
            start_time = time.time()
            
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=1024,  # Lower max tokens
                timeout=self.timeout,
                options=payload_options
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Add to history for reference
            self._add_message("user", user_input)
            self._add_message("assistant", response)
            
            # Calculate metrics
            tokens_per_second = len(response) / 4 / elapsed if elapsed > 0 else 0
            timestamp = time.strftime("%H:%M:%S")
            
            print(f"\nAI: {response}")
            print(f"\n[{timestamp} | {elapsed:.2f}s | ~{tokens_per_second:.1f} tokens/s]")
        
        # Restore original model
        self.model = original_model