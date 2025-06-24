import os
import sys
from pathlib import Path

def find_tesseract():
    """Attempt to find Tesseract executable in common locations"""
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\ladfr\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        # Add any other paths you might have installed to
    ]
    
    # Check common paths
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # Otherwise return default and let Tesseract find it in PATH
    return 'tesseract'

def process_image(img_path: str) -> str:
    """Extract text from images using OCR"""
    try:
        from PIL import Image
        import pytesseract
        
        # Set path to tesseract executable
        tesseract_path = find_tesseract()
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # ---- START TEST ----
        try:
            print(f"DEBUG: Pytesseract using Tesseract at: {pytesseract.pytesseract.tesseract_cmd}")
            print(f"DEBUG: Tesseract version: {pytesseract.get_tesseract_version()}")
            print(f"DEBUG: Available languages: {pytesseract.get_languages(config='')}") # This checks tessdata
        except Exception as e_test:
            print(f"DEBUG: Error during Tesseract test: {e_test}")
        # ---- END TEST ----
            
        # Verify image file exists
        if not os.path.exists(img_path):
            return f"Error: Image file not found: {img_path}"
        
        # Try to get Tesseract version to check if it's working (this is somewhat redundant with the test above but good for the error message)
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            setup_instructions = """
Tesseract OCR Setup Instructions:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install with default options
3. Ensure the installation directory is in your PATH environment variable
4. Restart your terminal and the application

If Tesseract is already installed but not found, add this line to your img_handler.py:
pytesseract.pytesseract.tesseract_cmd = r'C:\\path\\to\\tesseract.exe'
"""
            return f"Error: Tesseract OCR not found or not working.\n{setup_instructions}"
        
        # Open image
        img = Image.open(img_path)
        
        # Process image with some preprocessing for better OCR results
        try:
            # For better results, convert to grayscale and apply some enhancements
            from PIL import ImageEnhance
            img = img.convert('L')  # Convert to grayscale
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Increase contrast
        except Exception:
            # If enhancement fails, continue with original image
            pass
        
        # Extract text
        text = pytesseract.image_to_string(img)
        
        if not text.strip():
            return f"No text detected in the image. The image '{os.path.basename(img_path)}' either doesn't contain readable text or the quality might be too low for OCR."
        
        return text
    except ImportError:
        return "Required libraries not found. Please install them with:\npip install pillow pytesseract"
    except Exception as e:
        return f"Error processing image: {str(e)}"

from typing import Optional

def query_image(client, image_path: str, query: str, model: str = "llava:7b", system_prompt: Optional[str] = None):
    """Ask a question about an image (visual understanding)"""
    try:
        # Use a streamlined list of best models
        models_to_try = [
            "llava:7b",                # Original LLaVa - proven to work well
            "optimized-llava:latest",  # Your optimized version
            "llava:7b-v1.5-q4_0"       # Backup if needed
        ]
        
        # First try the specified model if not in the list
        if model not in models_to_try:
            models_to_try.insert(0, model)
        
        last_error = None
        
        for m in models_to_try:
            try:
                print(f"\nTrying model: {m}")
                # Add options with system prompt if provided
                options = {}
                if system_prompt:
                    options["system_message"] = system_prompt
                
                response = client.generate_with_image(m, query, image_path, max_tokens=2048, options=options)
                
                # If response looks good, return it
                if not response.startswith("Error") and not response.startswith("ERROR"):
                    print(f"Success with model: {m}")
                    return response
                else:
                    print(f"Failed with model: {m}")
                    last_error = response
            except Exception as e:
                last_error = f"Error with {m}: {str(e)}"
                print(last_error)
                continue
                
        # If all models failed
        return f"Sorry, I couldn't analyze this image. Please try with a different image or question."
    except Exception as e:
        return f"Error analyzing image: {str(e)}"