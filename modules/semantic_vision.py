"""
Semantic Vision-Language Understanding Module using OpenCLIP.
Extracts zero-shot visual probabilities for celebration themes, viewpoint classification (front vs side),
visual number 1 presence, and crowd/human presence to complement OCR and geometry filters.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SemanticVisionModel:
    _instance = None
    _model = None
    _preprocess = None
    _tokenizer = None

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.model_name = model_name
        self.pretrained = pretrained
        self._load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if SemanticVisionModel._model is None:
            try:
                import open_clip
                logger.info(f"Loading OpenCLIP model ({self.model_name})...")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device="cpu"
                )
                model.eval()
                tokenizer = open_clip.get_tokenizer(self.model_name)

                SemanticVisionModel._model = model
                SemanticVisionModel._preprocess = preprocess
                SemanticVisionModel._tokenizer = tokenizer
                logger.info("OpenCLIP model loaded and cached successfully.")
            except Exception as e:
                logger.warning(f"Failed to load OpenCLIP model: {e}")
                SemanticVisionModel._model = None

    @property
    def is_available(self) -> bool:
        return SemanticVisionModel._model is not None

    def evaluate_scene(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Runs zero-shot semantic inference on decoration scenes.
        Returns probabilities and diagnostic findings for:
          - Event theme (first birthday, general birthday, wedding, haldi, etc.)
          - Viewpoint (front-facing vs side angle vs close-up)
          - Presence of number one / banner in decor
          - Presence of people in scene
        """
        if not self.is_available or pil_image is None:
            return {
                "available": False,
                "reason": "OpenCLIP model not loaded",
                "is_first_birthday_prob": 0.0,
                "is_front_view_prob": 0.0,
                "has_people_prob": 0.0,
                "has_number_one_prob": 0.0
            }

        try:
            preprocess = SemanticVisionModel._preprocess
            model = SemanticVisionModel._model
            tokenizer = SemanticVisionModel._tokenizer

            image_tensor = preprocess(pil_image).unsqueeze(0)

            # Query 1: Event Theme Classification
            theme_prompts = [
                "a first birthday party decoration with balloons and number one",
                "a birthday party celebration decoration with banner",
                "a wedding stage floral decoration with mandap",
                "a haldi ceremony yellow floral decoration",
                "a baby shower celebration decoration",
                "an anniversary party backdrop decoration",
                "an empty room or ordinary indoor background"
            ]

            # Query 2: Viewpoint Classification
            view_prompts = [
                "a straight front-facing photograph of an event decoration backdrop",
                "an angled side-view photograph of a decoration setup",
                "a close-up detail photograph of balloons or flowers",
                "an obstructed or unclear angle photo"
            ]

            # Query 3: Specific Number One in Decoration
            number_prompts = [
                "an event decoration featuring the number one or 1st birthday banner",
                "an event decoration with no number one or numerals visible"
            ]

            # Query 4: Human Presence
            human_prompts = [
                "an empty decoration backdrop with no people or humans",
                "a photograph of people or guests posing in front of decoration"
            ]

            with torch.no_grad():
                # Process Theme
                theme_tokens = tokenizer(theme_prompts)
                img_feat = model.encode_image(image_tensor)
                theme_feat = model.encode_text(theme_tokens)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                theme_feat /= theme_feat.norm(dim=-1, keepdim=True)
                theme_probs = (100.0 * img_feat @ theme_feat.T).softmax(dim=-1)[0].cpu().numpy()

                # Process View
                view_tokens = tokenizer(view_prompts)
                view_feat = model.encode_text(view_tokens)
                view_feat /= view_feat.norm(dim=-1, keepdim=True)
                view_probs = (100.0 * img_feat @ view_feat.T).softmax(dim=-1)[0].cpu().numpy()

                # Process Number One
                num_tokens = tokenizer(number_prompts)
                num_feat = model.encode_text(num_tokens)
                num_feat /= num_feat.norm(dim=-1, keepdim=True)
                num_probs = (100.0 * img_feat @ num_feat.T).softmax(dim=-1)[0].cpu().numpy()

                # Process Human
                human_tokens = tokenizer(human_prompts)
                human_feat = model.encode_text(human_tokens)
                human_feat /= human_feat.norm(dim=-1, keepdim=True)
                human_probs = (100.0 * img_feat @ human_feat.T).softmax(dim=-1)[0].cpu().numpy()

            best_theme_idx = int(np.argmax(theme_probs))
            best_theme_name = theme_prompts[best_theme_idx]

            best_view_idx = int(np.argmax(view_probs))
            view_map = ["FRONT", "SIDE", "CLOSEUP", "OBSTRUCTED"]
            predicted_view = view_map[best_view_idx]

            return {
                "available": True,
                "top_theme": best_theme_name,
                "top_theme_prob": round(float(theme_probs[best_theme_idx]), 3),
                "is_first_birthday_prob": round(float(theme_probs[0]), 3),
                "is_general_birthday_prob": round(float(theme_probs[0] + theme_probs[1]), 3),
                "predicted_view": predicted_view,
                "is_front_view_prob": round(float(view_probs[0]), 3),
                "has_number_one_prob": round(float(num_probs[0]), 3),
                "has_people_prob": round(float(human_probs[1]), 3)
            }
        except Exception as e:
            logger.error(f"SemanticVision inference failed: {e}")
            return {
                "available": False,
                "error": str(e),
                "is_first_birthday_prob": 0.0,
                "is_front_view_prob": 0.0,
                "has_people_prob": 0.0,
                "has_number_one_prob": 0.0
            }
