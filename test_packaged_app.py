import os
import sys
import subprocess
import platform

def print_section(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)

# Get paths
exe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "Ladbon AI Desktop.exe")
dist_dir = os.path.dirname(exe_path)

print_section("Testing packaged application")
print(f"Executable path: {exe_path}")
print(f"Exists: {os.path.exists(exe_path)}")
print(f"File size: {os.path.getsize(exe_path) / (1024*1024):.2f} MB")
print(f"Created: {os.path.getctime(exe_path)}")

print_section("Environment")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Working directory: {os.getcwd()}")

print_section("Directory contents")
print(f"Contents of {dist_dir}:")
for item in os.listdir(dist_dir):
    item_path = os.path.join(dist_dir, item)
    size = os.path.getsize(item_path) / 1024  # KB
    print(f"  - {item} ({size:.2f} KB)")

# Try to run the executable with --diagnose flag
print_section("Running executable with diagnostics")
try:
    print("Starting executable...")
    # Use subprocess.Popen to run the executable without blocking
    process = subprocess.Popen([exe_path, "--diagnose"], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE,
                             text=True)
    print("Process started")
    
    # Give it a few seconds to start
    import time
    time.sleep(5)  # Wait 5 seconds
    
    # Check if process is still running
    if process.poll() is None:
        print("Process is still running after 5 seconds")
        # Try to terminate it
        process.terminate()
        print("Sent termination signal")
    else:
        stdout, stderr = process.communicate()
        print(f"Process exited with code {process.returncode}")
        if stdout:
            print("\nSTDOUT:")
            print(stdout[:1000])  # Print only first 1000 chars
        if stderr:
            print("\nSTDERR:")
            print(stderr[:1000])  # Print only first 1000 chars
except Exception as e:
    print(f"Error running executable: {e}")

print_section("Test complete")
