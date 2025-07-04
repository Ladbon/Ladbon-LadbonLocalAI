import os
import json
from cli.chat import ChatSession
from utils.ollama_client import OllamaClient
from PyQt5.QtWidgets import QInputDialog
from utils.data_paths import get_settings_path, first_run_migration
import init_llama # Import the new module

def load_settings():
    """Load settings from settings.json"""
    settings_path = get_settings_path()  # Use our utility function
    default_settings = {
        "model": "qwen3:8b", 
        "max_tokens": 8192,
        "custom_system_prompt": "",
        "timeout": 0
    }
    
    try:
        if os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                return settings
    except Exception as e:
        print(f"Error loading settings: {e}")
    
    return default_settings

def main():
    # Initialize the llama.cpp backend first
    if not init_llama.initialize_llama_backend():
        # The application should not proceed if the backend fails to initialize.
        # You might want to show an error message to the user.
        print("Fatal: Could not initialize the llama.cpp backend. Exiting.")
        # In a GUI app, you would show a message box.
        # For now, we'll just exit.
        return

    # Check for first run and migrate data if needed
    first_run_migration()
    
    # Load settings
    settings = load_settings()
    
    # Initialize Ollama client and chat session with settings
    client = OllamaClient()
    session = ChatSession(
        client=client, 
        model=settings.get("model", "qwen3:8b"),
        max_tokens=settings.get("max_tokens", 8192),
        timeout=settings.get("timeout", 0),
        system_prompt=settings.get("custom_system_prompt", None)
    )

    while True:
        choice = session.menu()
        if choice == "Exit":
            print("Goodbye!")
            break
        session.handle(choice)

if __name__ == "__main__":
    main()