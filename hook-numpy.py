"""
PyInstaller hook for NumPy to prevent CPU dispatcher initialization issues.
"""
from PyInstaller.utils.hooks import collect_all

# Get all of NumPy's files, not just the top level ones
datas, binaries, hiddenimports = collect_all('numpy')

# This is used to filter out specific modules that might cause issues
def filter_out_problematic(entry):
    # Just an example filter - you can customize this if specific modules cause issues
    problematic = ['numpy.core.multiarray_tests', 'numpy.core._multiarray_umath']
    return not any(p in entry for p in problematic)

# Filter hiddenimports if needed
# hiddenimports = list(filter(filter_out_problematic, hiddenimports))

# Add the hook variables that PyInstaller looks for
# This tells PyInstaller to include all of NumPy's data, binaries, and hidden imports
