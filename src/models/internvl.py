from models.model import Model
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import logging
import torch
import torchvision.transforms as T

logger = logging.getLogger(__name__)

class InternVL(Model):

    def __init__(self, model_config):
        super().__init__(model_config)
    
    def load(self):
        try:
            self.processor = AutoTokenizer.from_pretrained(
                self.model_config['path'],
                trust_remote_code = True
            )

            self.model = AutoModel.from_pretrained(
                self.model_config['path'],
                dtype = torch.bfloat16,
                trust_remote_code = True
            ).to(self.device).eval()

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
        
    def _build_transform(self, input_size):
        """Build image transformation pipeline"""
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation = InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
        ])
        return transform
    
    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _dynamic_preprocess(self, image, min_num = 1, max_num = 12, image_size = 448, use_thumbnail = False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def _load_image(self, image, input_size = 448, max_num = 12):
        image = image.convert('RGB')
        transform = self._build_transform(input_size = input_size)
        images = self._dynamic_preprocess(image, image_size = input_size, use_thumbnail = True, max_num = max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def generate(self, image, text):
        try:
            image = self._load_image(
                image
            ).to(torch.bfloat16).to(self.device)

            conversation = f"<image>\n{text}"

            generation_config = dict(
                max_new_tokens = 1024,
                do_sample = True,
                temperature = 0.7
            )

            with torch.inference_mode():
                response, history = self.model.chat(
                    self.processor, 
                    image, 
                    conversation, 
                    generation_config,
                    history = None,
                    return_history = True
                )

            self.history = history

            return response

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
        
    def continue_generate(self, text):
        try:
            generation_config = dict(
                max_new_tokens = 1024,
                do_sample = True,
                temperature = 0.7
            )
            
            with torch.inference_mode():
                response, history = self.model.chat(
                    self.processor,
                    None,
                    text,
                    generation_config,
                    history = self.history,
                    return_history = True
                )
            
            self.history = history

            return response

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
    
    def unload(self):
        return super().unload()