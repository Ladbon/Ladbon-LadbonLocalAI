# Ladbon AI Desktop

A desktop application for interacting with AI models through both local GGUF models (via llama-cpp-python) and Ollama, featuring document analysis, image recognition, web search capabilities, and RAG functionality.

## Features

- **Dual Model Support**: Use local GGUF models or Ollama models
- **Local Inference**: Run AI models directly on your computer with llama-cpp-python
- **HuggingFace Integration**: Download models directly from HuggingFace
- **Multi-Model Support**: Compatible with Llama, Mistral, Qwen, LLaVA, Gemma, Phi-3, and DeepSeek models
- **Document Analysis**: Process PDFs, TXT, and Word documents
- **Image Recognition**: Analyze images with compatible vision models
- **OCR Capabilities**: Extract text from images
- **Web Search**: Search the internet for real-time information
- **RAG Support**: Retrieve and generate answers based on your documents
- **Conversation Memory**: Maintains context for natural conversations
- **Settings Persistence**: Remembers your model selection and token preferences

## Requirements

```
PyQt5>=5.15.6
requests>=2.28.0
beautifulsoup4>=4.11.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
numpy>=1.20.0
Pillow>=9.0.0
PyPDF2>=2.0.0
python-docx>=0.8.11
pytesseract>=0.3.9
huggingface_hub>=0.16.0
tqdm>=4.65.0
psutil>=5.9.0
llama-cpp-python==0.3.9 (installed separately)
```

## Installation

### Prerequisites

1. **Install Python 3.8+** if not already installed (only needed if running from source)
2. **Optional: Install Ollama** from [ollama.com/download](https://ollama.com/download) if you want to use Ollama models in addition to local GGUF models

### Option 1: Windows Installer (Recommended)

1. Download the latest installer (`Ladbon_AI_Desktop_Setup.exe`) from the Releases page
2. Run the installer and follow the on-screen instructions
3. Launch Ladbon AI Desktop from your Start menu or desktop shortcut

### Option 2: Run from Source

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/localai.git
   cd localai
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install llama-cpp-python with proper configuration:
   ```
   # For NVIDIA GPU support:
   cd src
   python install_llamacpp.py
   
   # For CPU-only support:
   cd src
   python install_cpu_llamacpp.py
   ```

5. Run the application:
   ```
   cd src
   python gui_app.py
   ```

### Option 3: Portable Executable

1. Download the standalone executable (`Ladbon AI Desktop.exe`) from the Releases page
2. Create a folder where you want to run the application
3. Create `docs`, `img`, and `models` subfolders in the same location
4. Download GGUF models into the `models` folder (optional, for local inference)
5. Run `Ladbon AI Desktop.exe`

## Usage

### Getting Started

1. **Launch Ladbon AI Desktop**
2. **Choose your model source**:
   - **Local Models**: Select from available downloaded GGUF models
   - **Ollama Models**: Start Ollama on your system and select from available models
3. **Select a model** from the dropdown menu
4. **Type a message** and press Enter or click Send

### Working with Documents

1. Check the documents you want to include in the sidebar
2. Ask questions about the selected documents

### Image Analysis

1. Check the image you want to analyze in the sidebar
2. Select a compatible vision model (LLaVA, Llama 4, Phi-3)
3. Ask questions about the selected image

### Web Search

1. Click the **🔍 Web** button (turns green when active)
2. Enter your query
3. The AI will search the web and use the results in its response

### Adjusting Settings

1. Go to the **Settings** tab
2. Set the **maximum tokens** for responses in the input box
3. Click **Save Settings** to remember your preferences

## Customization

- **Add documents**: Place documents in the `docs` folder
- **Add images**: Place images in the `img` folder
- **Add models**: Place GGUF model files in the `models` folder
- **Logs**: View application logs in the **Logs** tab or in the `logs` folder

## Building the Application

### Creating the Executable

To create a standalone executable:

```
cd src
# First install llama-cpp-python if you want local model support
python install_llamacpp.py
# Then build the package
python package.py
```

The executable will be created in the `dist` folder. If llama-cpp-python is installed, it will be included in the package, enabling local model support.

### Creating an Installer

1. Run `package.py` to create the executable
2. Install Inno Setup from [jrsoftware.org/isinfo.php](https://jrsoftware.org/isinfo.php)
3. Open `LadboAIDesktop.iss` in Inno Setup
4. Press F9 or click Build > Compile
5. The installer will be created in the `installer` folder

## Known Limitations

- Ollama models require Ollama to be installed separately and running
- Local GGUF models require sufficient RAM based on model size
- GPU acceleration requires NVIDIA GPU with CUDA support
- Web search may be limited for complex queries
- Image analysis requires vision-capable models
- Some models have specific capabilities as noted in the UI

## Troubleshooting

- **No Ollama models appear**: Make sure Ollama is running (`ollama serve` in terminal)
- **No local models appear**: Check that you have GGUF models in the `models` directory
- **Error loading models**: Check your internet connection
- **Local model errors**: Run `python test_model_loading.py` to diagnose issues
- **Slow local inference**: Try using a smaller model or enabling GPU acceleration
- **llama-cpp-python errors**: Run `python reinstall_llamacpp.py` to reinstall
- **Image analysis not working**: Verify you've selected a vision-capable model
- **Slow responses**: Larger models require more processing time and resources
- **No logs displayed**: Check the `logs` folder to ensure log files are being created

## License

Apache 2.0

---

*Developed by Ladbon Fragari*