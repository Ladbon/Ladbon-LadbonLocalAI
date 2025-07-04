"""
Environment checking utility for Ladbon AI Desktop
Verifies system architecture, installation paths, and other critical settings
"""
import os
import sys
import platform
import ctypes
import logging

logger = logging.getLogger("environment_check")

def verify_environment():
    """
    Verify the system environment is suitable for running Ladbon AI Desktop
    Returns a dictionary with diagnosis results
    """
    results = {
        "is_64bit_python": sys.maxsize > 2**32,
        "is_64bit_os": platform.machine().endswith('64'),
        "python_version": platform.python_version(),
        "system_version": platform.version(),
        "installation_path": None,
        "models_path": None,
        "dlls_found": False,
        "dll_path_set": False,
        "issues": []
    }
    
    # Check if running in 32-bit Python
    if not results["is_64bit_python"]:
        results["issues"].append("CRITICAL: Running in 32-bit Python. llama-cpp-python requires 64-bit Python.")
    
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        exe_path = sys.executable
        results["installation_path"] = os.path.dirname(exe_path)
        logger.info(f"Running from frozen executable: {exe_path}")
        
        # Check if installed to Program Files (x86)
        if "Program Files (x86)" in results["installation_path"]:
            results["issues"].append("CRITICAL: Application is installed in Program Files (x86), which suggests 32-bit mode. "
                                    "Reinstall using the updated installer which enforces 64-bit installation.")
    else:
        results["installation_path"] = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Running as script from: {results['installation_path']}")
    
    # Check for user data directory
    if platform.system() == "Windows":
        expected_data_dir = os.path.expandvars(r"%LOCALAPPDATA%\Ladbon AI Desktop")
        results["models_path"] = os.path.join(expected_data_dir, "models")
        if not os.path.exists(expected_data_dir):
            results["issues"].append(f"User data directory not found at {expected_data_dir}")
    
    # Check for DLLs
    if platform.system() == "Windows":
        # Check if DLLs are in PATH
        path_entries = os.environ.get('PATH', '').split(os.pathsep)
        has_dll_path = False
        for path in path_entries:
            if 'llama_cpp\\lib' in path or 'llama_cpp/lib' in path:
                has_dll_path = True
                results["dll_path_set"] = True
                break
        
        if not has_dll_path:
            results["issues"].append("No llama_cpp/lib directory found in PATH environment variable")
        
        # Look for llama.dll in various locations
        dll_locations = []
        if getattr(sys, 'frozen', False):
            app_path = os.path.dirname(sys.executable)
            dll_locations.append(os.path.join(app_path, '_internal', 'llama_cpp', 'lib', 'llama.dll'))
            dll_locations.append(os.path.join(app_path, 'llama_cpp', 'lib', 'llama.dll'))
        else:
            try:
                import llama_cpp
                llama_cpp_dir = os.path.dirname(llama_cpp.__file__)
                dll_locations.append(os.path.join(llama_cpp_dir, 'lib', 'llama.dll'))
            except ImportError:
                pass
        
        # Check each location
        for location in dll_locations:
            if os.path.exists(location):
                results["dlls_found"] = True
                logger.info(f"Found llama.dll at {location}")
                
                # Try to check DLL architecture
                try:
                    # A hacky but simple way to check if a PE file is 64-bit
                    with open(location, 'rb') as f:
                        f.seek(0x3C)  # Offset to PE header pointer
                        pe_offset = int.from_bytes(f.read(4), byteorder='little')
                        f.seek(pe_offset + 4)  # Signature + Machine
                        machine = int.from_bytes(f.read(2), byteorder='little')
                        is_64bit_dll = machine == 0x8664  # IMAGE_FILE_MACHINE_AMD64
                        results["dll_is_64bit"] = is_64bit_dll
                        
                        if not is_64bit_dll:
                            results["issues"].append(f"CRITICAL: Found 32-bit llama.dll at {location}. 64-bit version required!")
                except Exception as e:
                    logger.error(f"Failed to check DLL architecture: {e}")
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = verify_environment()
    print("Environment check results:")
    for key, value in results.items():
        if key != "issues":
            print(f"  {key}: {value}")
    
    if results["issues"]:
        print("\nPotential issues found:")
        for issue in results["issues"]:
            print(f"  - {issue}")
