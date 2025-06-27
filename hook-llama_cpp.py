"""
PyInstaller hook for llama_cpp package
This ensures that all necessary files are included in the bundle
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
import os
import sys
import glob
import platform

print("=" * 60)
print("EXECUTING CUSTOM HOOK FOR LLAMA_CPP")
print("=" * 60)

# Collect all modules, binaries and data files
datas, binaries, hiddenimports = collect_all('llama_cpp')

# Add any specific hidden imports
hiddenimports += [
    'llama_cpp.llama_cpp',
    'llama_cpp._ctypes_extensions',
    'llama_cpp.lib_path',
    'ctypes',
    'pathlib',
    'importlib',
    'os',
    'sys',
]

# Ensure lib directory is included
import llama_cpp
llama_cpp_path = os.path.dirname(llama_cpp.__file__)
lib_path = os.path.join(llama_cpp_path, 'lib')
print(f"PyInstaller hook: Looking for llama_cpp lib directory at {lib_path}")

if os.path.exists(lib_path):
    print(f"PyInstaller hook: Found llama_cpp lib directory at {lib_path}")
    
    # First add the entire lib directory to make sure the directory structure exists
    datas.append((lib_path, os.path.join('llama_cpp', 'lib')))
    
    # Then handle individual files
    file_count = 0
    for file_path in glob.glob(os.path.join(lib_path, '**'), recursive=True):
        if os.path.isfile(file_path):
            file_count += 1
            # Calculate the appropriate destination path
            rel_path = os.path.relpath(file_path, llama_cpp_path)
            dest_path = os.path.join('llama_cpp', os.path.dirname(rel_path))
            print(f"PyInstaller hook: Adding binary {os.path.basename(file_path)} to {dest_path}")
            binaries.append((file_path, dest_path))
            
            # For Windows, also add DLLs to root directory for easier loading
            if platform.system() == "Windows" and file_path.endswith('.dll'):
                print(f"PyInstaller hook: Also adding DLL to root: {os.path.basename(file_path)}")
                binaries.append((file_path, '.'))
    
    print(f"PyInstaller hook: Added {file_count} files from llama_cpp/lib")
else:
    print(f"WARNING: llama_cpp lib directory not found at {lib_path}")
    # Try to handle the case where lib might be elsewhere
    # Check if we're using a different structure
    potential_lib_dirs = [
        os.path.join(os.path.dirname(llama_cpp_path), 'lib'),
        os.path.join(os.path.dirname(os.path.dirname(llama_cpp_path)), 'lib'),
    ]
    
    for potential_lib in potential_lib_dirs:
        if os.path.exists(potential_lib):
            print(f"PyInstaller hook: Found alternative lib directory at {potential_lib}")
            datas.append((potential_lib, 'lib'))
            
            for file_path in glob.glob(os.path.join(potential_lib, '**'), recursive=True):
                if os.path.isfile(file_path):
                    print(f"PyInstaller hook: Adding binary from alternative location: {file_path}")
                    binaries.append((file_path, 'lib'))

print("=" * 60)
print("FINISHED CUSTOM HOOK FOR LLAMA_CPP")
print("=" * 60)
