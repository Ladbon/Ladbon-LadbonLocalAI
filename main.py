from cli.chat import ChatSession
from utils.ollama_client import OllamaClient
from PyQt5.QtWidgets import QInputDialog

def main():
    # Initialize Ollama client and chat session
    client = OllamaClient()
    session = ChatSession(client=client, model="qwen3:8b", max_tokens=8192, timeout=None)

    while True:
        choice = session.menu()
        if choice == "Exit":
            print("Goodbye!")
            break
        session.handle(choice)

if __name__ == "__main__":
    main()