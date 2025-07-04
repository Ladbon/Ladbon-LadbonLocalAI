"""
PyInstaller hook for utils.llm_dll_fixer
"""
from PyInstaller.utils.hooks import collect_all

# Collect all modules, binaries and data files
datas, binaries, hiddenimports = collect_all('utils.llm_dll_fixer')

# Add specific hidden imports
hiddenimports.extend([
    'ctypes',
    'platform',
    'logging',
    'os',
    'sys',
])
