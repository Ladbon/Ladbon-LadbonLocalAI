import os
import requests
import tempfile
import subprocess
import json
import inspect
import traceback
from utils.logger import setup_logger
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import sys

logger = setup_logger('huggingface_manager')

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class HuggingFaceManager:
    """Improved Hugging Face integration for model discovery and download"""
    
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
        
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Available model templates - just identifiers and repos, not sizes
        self.model_templates = {
            "llama2-7b": {
                "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
                "filename": "llama-2-7b-chat.Q4_K_M.gguf",
                "description": "Llama 2 7B (Chat)",
            },
            "phi3-mini": {
                "repo_id": "TheBloke/phi-3-mini-4k-instruct-GGUF",
                "filename": "phi-3-mini-4k-instruct.Q4_K_M.gguf",
                "description": "Phi-3 Mini (Chat)",
            },
            "mistral-7b": {
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "description": "Mistral 7B Instruct v0.2",
            },
            "tinyllama": {
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "description": "TinyLlama 1.1B (Fast Chat)",
            }
        }
        
    # Complete the get_model_info method
    def get_model_info(self, model_id):
        """Get information about a model from Hugging Face without downloading"""
        # Find model template - first check predefined models
        if model_id in self.model_templates:
            model_template = self.model_templates[model_id]
        else:
            # Try to find in trending models and search results
            all_models = self.list_available_models(include_trending=True)
            
            if model_id in all_models:
                model_template = all_models[model_id]
                logger.info(f"Using non-predefined model info for: {model_id}")
            else:
                logger.error(f"Unknown model ID: {model_id}")
                return None

        try:
            # Use direct URL for HuggingFace CDN instead of API
            repo_id = model_template["repo_id"]
            filename = model_template["filename"]
            
            # Try direct download URL which is more reliable for size checks
            file_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            logger.info(f"Checking file at: {file_url}")
            
            import requests
            head_response = requests.head(file_url, allow_redirects=True)
            
            if head_response.status_code == 200:
                # We found the file
                size_bytes = int(head_response.headers.get('Content-Length', 0))
                size_mb = size_bytes / (1024 * 1024)
                
                # Create info dict
                info = {
                    "size": size_bytes,
                    "size_mb": size_mb,
                    "size_readable": f"{size_mb:.1f} MB",
                    "etag": head_response.headers.get('ETag', ''),
                    "url": file_url
                }
                
                # Combine with template info
                info.update(model_template)
                return info
            else:
                logger.error(f"Error accessing model file: {head_response.status_code}")
                
                # Try alternative branch name (some repos use 'master' instead of 'main')
                alt_url = f"https://huggingface.co/{repo_id}/resolve/master/{filename}"
                alt_response = requests.head(alt_url, allow_redirects=True)
                
                if alt_response.status_code == 200:
                    size_bytes = int(alt_response.headers.get('Content-Length', 0))
                    size_mb = size_bytes / (1024 * 1024)
                    
                    info = {
                        "size": size_bytes,
                        "size_mb": size_mb,
                        "size_readable": f"{size_mb:.1f} MB",
                        "etag": alt_response.headers.get('ETag', ''),
                        "url": alt_url
                    }
                    
                    info.update(model_template)
                    return info
                
                return None
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None
    
    def get_trending_models(self, limit=20):
        """Get trending GGUF models from Hugging Face Hub to match website list"""
        try:
            import requests
            import re
            
            # Use the same approach as the website - use combined models
            trending_data = []
            
            # First get trending models (most important)
            api_url = "https://huggingface.co/api/models?trending=1&filter=gguf&limit=40"
            logger.info(f"Fetching trending models from: {api_url}")
            
            response = requests.get(api_url)
            if response.status_code == 200:
                trending_data.extend(response.json())
                logger.info(f"Found {len(trending_data)} trending models")
            
            # If we need more, get high download models
            if len(trending_data) < 40:
                api_url = "https://huggingface.co/api/models?sort=downloads&direction=-1&filter=gguf&limit=40"
                logger.info(f"Fetching popular models from: {api_url}")
                
                response = requests.get(api_url)
                if response.status_code == 200:
                    for model in response.json():
                        # Add if not already in trending_data
                        model_id = model.get('id', '')
                        if model_id and not any(m.get('id', '') == model_id for m in trending_data):
                            trending_data.append(model)
            
                logger.info(f"Total models after adding downloads: {len(trending_data)}")
            
            # Process the results
            formatted_models = {}
            count = 0
            
            # First pass to identify models that have GGUF files
            for model in trending_data:
                try:
                    if count >= limit:
                        break
                    
                    repo_id = model.get('id', '')
                    if not repo_id or '/' not in repo_id:
                        continue
                    
                    model_name = repo_id.split('/')[-1]
                    safe_model_id = model_name.lower().replace("-", "_")
                    
                    # Skip some model types
                    lower_name = model_name.lower()
                    if any(s in lower_name for s in ['embedding', 'tokenizer', 'bert']):
                        continue
                    
                    # Get the repo's files to find GGUF files
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi()
                        
                        files = api.list_repo_files(repo_id)
                        gguf_files = [f for f in files if f.lower().endswith('.gguf')]
                        if not gguf_files:
                            continue
                        
                        # Find a reasonable sized file (prefer Q4_K_M quantization)
                        # Many repos have tons of files - we want a good default
                        q4_files = [f for f in gguf_files if re.search(r'q4_k_m|Q4_K_M', f)]
                        
                        if q4_files:
                            filename = min(q4_files, key=len)  # Get the shortest name
                        else:
                            # Try to find a reasonable sized file - avoid huge ones
                            small_files = [f for f in gguf_files if re.search(r'q4|Q4|4bit|4-bit', f)]
                            if small_files:
                                filename = min(small_files, key=len)
                            else:
                                filename = min(gguf_files, key=len)
                        
                        # Clean up description
                        description = model.get('description', 'No description')
                        if description:
                            # Take the first line, limit length
                            lines = description.split('\n')
                            description = lines[0][:150]
                            if len(description) >= 150:
                                description += "..."
                        
                        # Add to results
                        formatted_models[safe_model_id] = {
                            "repo_id": repo_id,
                            "filename": filename,
                            "description": description,
                            "downloads": model.get('downloads', 0),
                            "likes": model.get('likes', 0),
                            "tags": model.get('tags', []),
                            "pipeline_tag": model.get('pipeline_tag', ''),
                            "is_trending": True
                        }
                        count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing trending model {repo_id}: {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Error processing model data: {str(e)}")
                    continue
        
            logger.info(f"Successfully found {len(formatted_models)} trending GGUF models")
            return formatted_models
        
        except Exception as e:
            import traceback
            logger.error(f"Error fetching trending models: {str(e)}\n{traceback.format_exc()}")
            return {}
                
    def search_models(self, query, limit=20):
        """Search for GGUF models on Hugging Face Hub"""
        if not query or len(query.strip()) < 2:
            return {}
            
        try:
            # Create a temporary Python script to search for models
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
                script_path = temp.name
                temp.write(f"""
import json
import sys
from huggingface_hub import HfApi

try:
    # Initialize API
    api = HfApi()
    
    # Search for models
    models = api.list_models(
        filter="gguf",
        search="{query}",
        limit={limit*2},  # Request more to filter
        full=True
    )
    
    # Filter and format results
    results = []
    for model in models:
        # Only get models, not datasets or spaces
        if model.pipeline_tag in ["text-generation", "text2text-generation", "conversational"]:
            try:
                # Get files in the repo to find GGUF files
                files = api.list_repo_files(model.id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                
                # Skip if no GGUF files
                if not gguf_files:
                    continue
                    
                # Find a Q4_K_M.gguf file if possible
                q4_files = [f for f in gguf_files if "Q4_K_M" in f]
                filename = q4_files[0] if q4_files else gguf_files[0]
                
                result = {{
                    "repo_id": model.id,
                    "model_name": model.id.split("/")[-1] if "/" in model.id else model.id,
                    "author": model.id.split("/")[0] if "/" in model.id else "Unknown",
                    "description": model.description or "No description",
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "filename": filename,
                }}
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= {limit}:
                    break
            except Exception as inner_e:
                # Skip this model on error
                pass
    
    # Return the results
    print(json.dumps({{"results": results}}))
    
except Exception as e:
    print(json.dumps({{"error": str(e)}}), file=sys.stderr)
    sys.exit(1)
""")
            
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True
            )
            
            # Clean up the temp file
            os.unlink(script_path)
            
            if result.returncode != 0:
                logger.error(f"Error searching models: {result.stderr}")
                return {}
                
            try:
                # Parse the output
                data = json.loads(result.stdout)
                search_results = data.get("results", [])
                
                # Format the results
                formatted_results = {}
                for model in search_results:
                    model_id = model["model_name"].lower().replace("-", "_")
                    formatted_results[model_id] = {
                        "repo_id": model["repo_id"],
                        "filename": model["filename"],
                        "description": model["description"],
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                        "is_search_result": True
                    }
                    
                return formatted_results
                
            except json.JSONDecodeError:
                logger.error(f"Error parsing search results: {result.stdout}")
                return {}
        except Exception as e:
            logger.error(f"Error searching models: {str(e)}")
            return {}

    def list_available_models(self, include_trending=True):
        """List available models including trending ones"""
        # Start with predefined models
        models = {}
        
        # Add predefined models
        for model_id, template in self.model_templates.items():
            # Check if already downloaded
            local_path = os.path.join(self.models_dir, template["filename"])
            is_downloaded = os.path.exists(local_path)
            
            # Add basic info
            models[model_id] = {
                **template,
                "is_downloaded": is_downloaded,
                "local_path": local_path if is_downloaded else None,
                "is_predefined": True
            }
            
            # If downloaded, add file size
            if is_downloaded:
                size_bytes = os.path.getsize(local_path)
                models[model_id]["size_bytes"] = size_bytes
                models[model_id]["size_mb"] = size_bytes / (1024 * 1024)
        
        # Add trending models if requested
        if include_trending:
            try:
                trending = self.get_trending_models(limit=10)
                
                # Check if trending is a dictionary before using items()
                if isinstance(trending, dict):
                    for model_id, info in trending.items():
                        if model_id not in models:  # Don't overwrite predefined models
                            # Check if already downloaded
                            local_path = os.path.join(self.models_dir, info["filename"])
                            is_downloaded = os.path.exists(local_path)
                            
                            models[model_id] = {
                                **info,
                                "is_downloaded": is_downloaded,
                                "local_path": local_path if is_downloaded else None,
                                "is_trending": True
                            }
                            
                            # If downloaded, add file size
                            if is_downloaded:
                                size_bytes = os.path.getsize(local_path)
                                models[model_id]["size_bytes"] = size_bytes
                                models[model_id]["size_mb"] = size_bytes / (1024 * 1024)
                else:
                    logger.warning(f"Trending models not in expected format: {type(trending)}")
            except Exception as e:
                logger.error(f"Error loading trending models: {str(e)}")
        
        return models
    
    def download_model(self, model_id, progress_callback=None):
        """Download a model using hf_hub_download directly."""
        import os
        logger = setup_logger('model_manager')

        # Get all available models
        available_models = self.list_available_models()
        
        if model_id not in available_models:
            logger.error(f"Model ID {model_id} not found in available models.")
            return False, f"Model {model_id} not found in available models"
                
        model_info = available_models[model_id]
        os.makedirs(self.models_dir, exist_ok=True)
        
        output_path = os.path.join(self.models_dir, model_info["filename"])
        
        if os.path.exists(output_path):
            logger.info(f"Model {model_id} (file: {model_info['filename']}) already downloaded at {output_path}")
            return True, output_path
        
        try:
            logger.info(f"Downloading model: {model_id} from repo: {model_info['repo_id']} to file: {model_info['filename']}")
            
            # Using hf_hub_download directly
            downloaded_path = hf_hub_download(
                repo_id=model_info["repo_id"],
                filename=model_info["filename"],
                local_dir=self.models_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            if os.path.exists(downloaded_path):
                logger.info(f"Successfully downloaded {model_id} to {downloaded_path}")
                if progress_callback:
                    progress_callback(100)
                return True, downloaded_path
            else:
                logger.error(f"Download completed but file not found at {downloaded_path} for model {model_id}")
                return False, f"Download failed for {model_id}, file not found post-download."

        except Exception as e:
            logger.exception(f"Error downloading model {model_id}")
            return False, f"Error downloading {model_id}: {str(e)}"
    
    def simple_download_model(self, model_id, progress_callback=None, token=None):
        """Enhanced download method with better progress tracking"""
        import os
        import time
        
        logger.info(f"Starting download for model: {model_id}")
        
        try:
            # Get model information
            available_models = self.list_available_models(include_trending=True)
            if model_id not in available_models:
                logger.error(f"Model {model_id} not found in available_models.")
                return False, f"Model {model_id} not found"
            
            model_info = available_models[model_id]
            repo_id = model_info["repo_id"]
            filename = model_info["filename"]
            
            # Setup directories and output path
            os.makedirs(self.models_dir, exist_ok=True)
            output_path = os.path.join(self.models_dir, filename)
            
            # Signal start of download
            if progress_callback:
                progress_callback(0)
            
            # Custom progress callback that updates our UI
            class UIProgressCallback:
                def __init__(self, ui_callback):
                    self.ui_callback = ui_callback
                    self.last_percent = 0
                    self.last_update_time = time.time()
                
                def __call__(self, progress, total):
                    if total > 0:
                        percent = int(min(progress / total * 100, 100))
                        
                        # Only update UI if percent changed or 0.5 seconds passed
                        current_time = time.time()
                        if percent != self.last_percent or current_time - self.last_update_time > 0.5:
                            logger.info(f"Download progress: {percent}% ({progress/(1024*1024):.1f}/{total/(1024*1024):.1f} MB)")
                            if self.ui_callback:
                                # Use PyQt's thread-safe approach
                                from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                                QMetaObject.invokeMethod(
                                    self.ui_callback.__self__, 
                                    self.ui_callback.__name__, 
                                    Qt.ConnectionType.QueuedConnection,
                                    Q_ARG(int, percent)
                                )
                            self.last_percent = percent
                            self.last_update_time = current_time
            
            # Create our progress tracker
            ui_progress = UIProgressCallback(progress_callback) if progress_callback else None
            
            logger.info(f"Downloading model {model_id} from {repo_id}/{filename}")
            
            try:
                from huggingface_hub import hf_hub_download
                kwargs = {
                    "repo_id": repo_id,
                    "filename": filename,
                    "local_dir": self.models_dir,
                    "force_download": False,
                    "resume_download": True,
                }
                # Check if the file already exists
                if os.path.exists(output_path):
                    logger.info(f"Model {model_id} is already downloaded.")
                    if progress_callback:
                        progress_callback(100)
                    return True, output_path

                # Add token if provided
                if token:
                    kwargs["token"] = token
                    logger.info("Using authentication token for download")
                    
                # Check if this version supports progress_callback
                import inspect
                if 'progress_callback' in inspect.signature(hf_hub_download).parameters and ui_progress:
                    kwargs["progress_callback"] = ui_progress
                    logger.info("Using native progress callback")
                    
                # Download the model
                downloaded_path = hf_hub_download(**kwargs)
                
                if os.path.exists(downloaded_path):
                    logger.info(f"Successfully downloaded {model_id} to {downloaded_path}")
                    if progress_callback:
                        progress_callback(100)
                    return True, downloaded_path
                else:
                    logger.error(f"Download completed but file not found at expected path")
            except Exception as e:
                logger.warning(f"Error with huggingface_hub download: {str(e)}")
                # Fall back to direct download if needed
                
            # Signal completion if we reached this point
            if progress_callback:
                progress_callback(100)
                
            return True, output_path
            
        except Exception as e:
            logger.exception(f"Error downloading model {model_id}")
            return False, f"Error downloading {model_id}: {str(e)}"
    
    def check_huggingface_version(self):
        """Check HuggingFace Hub version and capabilities"""
        from PyQt5.QtWidgets import QMessageBox
        import inspect

        try:
            # Get version info
            import huggingface_hub
            version = huggingface_hub.__version__
            
            # Check hf_hub_download parameters
            from huggingface_hub import hf_hub_download
            params = inspect.signature(hf_hub_download).parameters
            param_names = list(params.keys())
            
            # Build info message
            info = f"HuggingFace Hub Version: {version}\n\n"
            info += "hf_hub_download parameters:\n"
            info += "\n".join([f"- {p}" for p in param_names])
            
            # Check if important parameters are available
            has_progress = 'progress_callback' in param_names
            has_force = 'force_download' in param_names
            has_resume = 'resume_download' in param_names
            
            info += f"\n\nSupports progress_callback: {has_progress}"
            info += f"\nSupports force_download: {has_force}"
            info += f"\nSupports resume_download: {has_resume}"
            
            # Show the information
            QMessageBox.information(None, "HuggingFace Hub Info", info)
            
            # Log this information
            logger.info(f"HuggingFace Hub Version: {version}")
            logger.info(f"hf_hub_download parameters: {', '.join(param_names)}")
            
            # If missing important parameters, suggest an upgrade
            if not has_progress or not has_force or not has_resume:
                upgrade_msg = ("Your version of huggingface_hub is missing some useful features.\n\n"
                              "Consider upgrading with:\n"
                              "pip install --upgrade huggingface_hub")
                QMessageBox.warning(None, "Consider Upgrading", upgrade_msg)
    
        except Exception as e:
            logger.error(f"Error checking HuggingFace version: {str(e)}")
            QMessageBox.critical(None, "Error", f"Could not check HuggingFace version: {str(e)}")