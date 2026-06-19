import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

_processor = None
_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Lazily load the BLIP captioning model from the local cache."""
    global _processor, _model
    if _processor is None or _model is None:
        model_id = "Salesforce/blip-image-captioning-large"
        print(f"[Vision] Loading {model_id} onto {device}...")
        _processor = BlipProcessor.from_pretrained(model_id, local_files_only=True)
        _model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=True).to(device)
        print("[Vision] BLIP model loaded successfully.")

def get_image_descriptions(image_path: str) -> dict:
    """Generate 5 diverse visual descriptions of the image at image_path."""
    load_model()
    
    if not os.path.exists(image_path):
        print(f"[Vision] Warning: Image path does not exist: {image_path}")
        return {
            "unconditional": "Image file not found.",
            "close_up": "Image file not found.",
            "main_object": "Image file not found.",
            "object_part": "Image file not found.",
            "damage_visible": "Image file not found."
        }
        
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[Vision] Error loading image {image_path}: {e}")
        return {
            "unconditional": f"Error loading image: {str(e)}",
            "close_up": f"Error loading image: {str(e)}",
            "main_object": f"Error loading image: {str(e)}",
            "object_part": f"Error loading image: {str(e)}",
            "damage_visible": f"Error loading image: {str(e)}"
        }

    prefixes = {
        "unconditional": "",
        "close_up": "a close-up photo of",
        "main_object": "the main object in this photo is a",
        "object_part": "the part of the object shown is the",
        "damage_visible": "the damage visible on this object is"
    }
    
    results = {}
    with torch.no_grad():
        for key, prefix in prefixes.items():
            try:
                if prefix:
                    inputs = _processor(raw_image, prefix, return_tensors="pt").to(device)
                    out = _model.generate(**inputs, max_new_tokens=40)
                    desc = _processor.decode(out[0], skip_special_tokens=True).strip()
                else:
                    inputs = _processor(raw_image, return_tensors="pt").to(device)
                    out = _model.generate(**inputs, max_new_tokens=40)
                    desc = _processor.decode(out[0], skip_special_tokens=True).strip()
                results[key] = desc
            except Exception as e:
                print(f"[Vision] Error generating caption for {key} with image {image_path}: {e}")
                results[key] = f"Error: {str(e)}"
                
    return results
