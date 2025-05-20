GitHub Copilot: # Ladbon AI Desktop

A desktop application for interacting with AI models through Ollama, featuring document analysis, image recognition, and web search capabilities.

## Features

- **Multi-Model Support**: Compatible with Llama, Mistral, Qwen, LLaVA, Gemma, Phi-3, and DeepSeek models
- **Document Analysis**: Process PDFs, TXT, and Word documents
- **Image Recognition**: Analyze images with compatible vision models
- **OCR Capabilities**: Extract text from images
- **Web Search**: Search the internet for real-time information
- **Conversation Memory**: Maintains context for natural conversations
- **Settings Persistence**: Remembers your model selection and token preferences

## Requirements

```
PyQt5>=5.15.0
requests>=2.25.0
beautifulsoup4>=4.9.0 
pillow>=8.0.0
PyPDF2>=2.0.0
python-docx>=0.8.10
pytesseract>=0.3.8
```

## Installation

### Prerequisites

1. **Install Ollama** from [ollama.com/download](https://ollama.com/download)
2. **Install Python 3.8+** if not already installed

### Option 1: Run from Source

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

4. Run the application:
   ```
   cd src
   python gui_app.py
   ```

### Option 2: Run Executable

1. Download the latest release from the Releases page
2. Extract the ZIP file
3. Run `Ladbon AI Desktop.exe`

## Usage

### Getting Started

1. **Start Ollama** on your system
2. **Launch Ladbon AI Desktop**
3. **Select a model** from the dropdown menu
4. **Type a message** and press Enter or click Send

### Working with Documents

1. Click the **Documents** tab
2. Check the documents you want to include
3. Ask questions about the selected documents

### Image Analysis

1. Click the **Images** tab
2. Check the image you want to analyze
3. Select a compatible vision model (LLaVA, Llama 4, Phi-3)
4. Ask questions about the selected image

### Web Search

1. Click the **🔍 Web** button (turns green when active)
2. Enter your query
3. The AI will search the web and use the results in its response

### Adjusting Settings

1. Set the **maximum tokens** for responses in the input box
2. Click **Save Settings** to remember your preferences

## Customization

- **Add documents**: Place documents in the docs folder
- **Add images**: Place images in the img folder
- **Logs**: View application logs in the logs folder

## Packaging the Application

To create a standalone executable:

```
cd src
python package.py
```

The executable will be created in the `dist` folder.

## Known Limitations

- Requires Ollama to be installed separately and running
- Web search may be limited for complex queries
- Image analysis requires vision-capable models
- Some models have specific capabilities as noted in the UI

## Troubleshooting

- **No models appear**: Make sure Ollama is running (`ollama serve` in terminal)
- **Error loading models**: Check your internet connection
- **Image analysis not working**: Verify you've selected a vision-capable model
- **Slow responses**: Larger models require more processing time and resources

## License

Apache 2.0

---

*Developed by Ladbon Fragari*