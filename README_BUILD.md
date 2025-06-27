# Ladbon AI Desktop

A desktop application for running local AI models and connecting to Ollama.

## Building the Application

The application can be built in two different modes:

### Directory Mode (default)

The directory mode builds the application as a directory structure. This is the recommended approach as it provides better performance and reliability.

```bash
python package.py
```

The resulting application will be in `dist/Ladbon AI Desktop/`

### Single Executable Mode

The single executable mode packages everything into a single `.exe` file. This is more convenient for distribution but may have issues with DLL loading.

```bash
python package.py --onefile
```

The resulting executable will be `dist/Ladbon AI Desktop.exe`

## Running the Application

### From Directory Mode

Run the `Ladbon AI Desktop.exe` file in the `dist/Ladbon AI Desktop/` directory.

### From Single Executable Mode

Run the `dist/Ladbon AI Desktop.exe` file directly.

## Common Issues

### Missing llama_cpp/lib Directory

If you encounter this error when using the single executable:

```
FileNotFoundError: [WinError 3] Det går inte att hitta sökvägen: '...\_MEI...\llama_cpp\lib'
```

Try using the directory mode build instead, which handles the library structure properly.

### CPU Dispatcher Tracer Already Initialized

This error occurs when NumPy's CPU dispatcher is initialized multiple times. The application includes fixes to prevent this, but if it still occurs:

1. Use the directory mode build
2. Check the logs for more detailed information (`numpy_hook_*.log` and `numpy_fix_debug_*.log`)

## Installing Models

### Local Models

Local GGUF models should be placed in the `models` directory inside your application folder.

### Ollama Models

Ollama models are automatically detected from your Ollama installation. Make sure Ollama is running before starting the application.

## Log Files

Log files are stored in the `logs` directory inside the application folder. If you encounter any issues, check these logs for detailed information.

## Dependencies

- Python 3.9 or newer
- llama-cpp-python
- PyQt5
- Ollama (optional)
