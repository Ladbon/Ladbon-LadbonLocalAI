
from PyInstaller.utils.hooks import copy_metadata

# Ensure the init_llama module is included
datas = copy_metadata('init_llama')
