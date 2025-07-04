"""
PyInstaller hook for utils.dll_loader module
"""
from PyInstaller.utils.hooks import collect_all

# This will ensure utils.dll_loader is properly bundled
datas, binaries, hiddenimports = collect_all('utils.dll_loader')

# Add the module to hidden imports
hiddenimports += ['utils.dll_loader']
