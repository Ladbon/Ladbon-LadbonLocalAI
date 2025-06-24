import importlib, ctypes, llama_cpp
llama_cpp.llama_cpp._lib = importlib.import_module("llama_cpp.llama_cpp")._lib
llama_cpp.llama_cpp._lib.llama_backend_init.argtypes = [ctypes.c_bool]
llama_cpp.llama_cpp._lib.llama_backend_init.restype  = None
def backend_init(numa: bool = False):
    return llama_cpp.llama_cpp._lib.llama_backend_init(ctypes.c_bool(numa))
llama_cpp.llama_backend_init = backend_init  # Fixed: removed trailing comma

import sys
import os
from PyQt5.QtWidgets import QApplication
from api.app import LocalAIApp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gui_app')

def main():
    # Initialize the application
    app = QApplication(sys.argv)
    window = LocalAIApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    print("INFO_MAIN: Starting main application (gui_app.py)...") 
    main()