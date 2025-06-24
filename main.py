import os
import json
from cli.chat import ChatSession
from utils.ollama_client import OllamaClient
from PyQt5.QtWidgets import QInputDialog

def load_settings():
    """Load settings from settings.json"""
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')
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