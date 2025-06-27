"""
Type stubs for PyInstaller-specific attributes.
This file helps IDEs understand PyInstaller-specific attributes like sys._MEIPASS.
"""
import sys
import os
from typing import Dict, Any, List, Optional, Callable, Union

# Add PyInstaller-specific attributes to standard modules
if not hasattr(sys, '_MEIPASS'):
    # Only define in dev environment, not at runtime
    sys._MEIPASS: str = ""  # type: ignore

# Helper functions that can be imported and used in other modules
def get_bundle_dir() -> str:
    """Get the PyInstaller bundle directory or current directory if not in a bundle"""
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

def is_bundled() -> bool:
    """Check if application is running from a PyInstaller bundle"""
    return hasattr(sys, '_MEIPASS')

def get_resource_path(relative_path: str) -> str:
    """Get absolute path to a resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# For other dynamic module patching
class DynamicModule:
    """Stub for dynamically created modules"""
    __cpu_features__: Dict[str, Any]
    __cpu_baseline__: List[str]
    __cpu_dispatch__: List[str]
    
    def get_cpu_features(self) -> Dict[str, bool]:
        """Stub for CPU feature detection function"""
        return {}
    
    def implement_cpu_features(self, *args: Any, **kwargs: Any) -> None:
        """Stub for CPU feature implementation function"""
        pass
