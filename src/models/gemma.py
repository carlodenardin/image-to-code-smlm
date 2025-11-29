from models.model import Model
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import logging
import torch

logger = logging.getLogger(__name__)

class Gemma(Model):

    def __init__(self, model_config):
        super().__init__(model_config)
    
    def load(self):
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_config['path'],
                use_fast = True
            )

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_config['path'],
                dtype = torch.bfloat16
            ).to(self.device).eval()

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
    
    def _generate_helper(self, inputs):
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = 1024,
                do_sample = True,
                temperature = 0.7
            )
        
        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens = True
        )

        self.history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        return response

    def generate(self, image, text):
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ]

            self.history = conversation.copy()

            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt = True,
                tokenize = True,
                return_dict = True,
                return_tensors = "pt"
            ).to(self.device, dtype = torch.bfloat16)

            return self._generate_helper(inputs)

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
        
    def continue_generate(self, text):
        try:
           
            self.history.append({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })

            inputs = self.processor.apply_chat_template(
                self.history,
                add_generation_prompt = True,
                tokenize = True,
                return_dict = True,
                return_tensors = "pt"
            ).to(self.device, dtype = torch.bfloat16)

            return self._generate_helper(inputs)

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
    
    def unload(self):
        return super().unload()