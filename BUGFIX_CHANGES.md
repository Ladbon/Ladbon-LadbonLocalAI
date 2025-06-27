# Bug Fixes - June 26, 2025

This update addresses several critical bugs in the LocalAI Desktop application:

## 1. Fixed Thread Stuck in CLI Chat

**Problem:** The CLI chat in `cli/chat.py` would get stuck indefinitely when there were connection issues with Ollama.

**Fix:** Added a default timeout of 60 seconds to prevent the thread from getting stuck permanently. This ensures the application remains responsive even if Ollama is unresponsive.

```python
# Changed in __init__ method in cli/chat.py:
self.timeout = timeout  # Default to 60 seconds instead of no timeout
```

## 2. Fixed Log Encoding Issues

**Problem:** Error reading logs with message: `'charmap' codec can't decode byte 0x8f in position 1800: character maps to <undefined>`

**Fix:** Added UTF-8 encoding parameter to the FileHandler in gui_app.py:

```python
logging.FileHandler(log_file, encoding='utf-8')
```

## 3. Fixed Model Download Progress Reporting

**Problem:** The model download progress was showing in terminal but not updating in the UI.

**Fix:** 
- Fixed the HuggingFaceManager initialization to correctly use the ModelDownloader
- Updated the download function in api/app.py to properly use the threaded downloader
- Improved thread-safety of the progress callback functions using QTimer.singleShot

```python
# First try to use the new downloader if available
if hasattr(self.huggingface_manager, 'downloader') and self.huggingface_manager.downloader:
    # Use improved downloader with progress callback
    ...
```

## 4. Improved Error Handling for Missing Models

**Problem:** The application would crash with `'NoneType' object has no attribute 'generate'` when Ollama was unavailable.

**Fix:** The OllamaClient.generate method now returns a user-friendly error message instead of crashing when Ollama is unavailable.

## 5. Code Organization

- Fixed variable scope issues in huggingface_manager.py where `has_downloader` was referenced before definition
- Improved error handling throughout the codebase
- Added more progress feedback and logging during downloads

These changes make the application more robust and provide better user feedback during downloads and when errors occur.
