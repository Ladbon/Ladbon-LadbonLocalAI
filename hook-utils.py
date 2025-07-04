"""
PyInstaller hook for utils module
"""
from PyInstaller.utils.hooks import collect_submodules, collect_all

# Collect all submodules from utils package
hiddenimports = collect_submodules('utils')

# Also collect all the modules individually for good measure
for module in hiddenimports:
    datas, binaries, more_imports = collect_all(module)
