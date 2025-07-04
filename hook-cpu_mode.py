"""
PyInstaller hook for ensuring all CPU-only mode code is included
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Make sure our new utility modules are included
hiddenimports = [
    'utils.cuda_detection',
    'utils.llamacpp_client_patched',
]

# Make sure our new data files are included
datas = []

# Ensure new command-line argument is recognized
def patch_app_code(app):
    """
    Add code to the packaged app to handle CPU-only mode argument
    """
    app_code = """
# CPU-only mode initialization added by hook-cpu_mode.py
import os
import sys
import logging

# Check if --force-cpu is in command line args
if '--force-cpu' in sys.argv:
    os.environ['FORCE_CPU_ONLY'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['DISABLE_LLAMA_CPP_CUDA'] = '1'
    
    # Try to set up logging early
    try:
        logger = logging.getLogger('cpu_mode_hook')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("CPU-only mode enabled via command line")
    except Exception:
        pass
        
    # Try to import our CUDA detection module early
    try:
        import utils.cuda_detection
    except Exception:
        pass
    """
    
    return app.add_runtime_hooks_after_entries([app_code])
