from models.model import Model
from transformers import AutoTokenizer, AutoModel

import logging
import torch

logger = logging.getLogger(__name__)

class MiniCPM(Model):

    def __init__(self, model_config):
        super().__init__(model_config)
    
    def load(self):

        print(self.model_config['path'])
        try:
            self.processor = AutoTokenizer.from_pretrained(
                self.model_config['path'],
                trust_remote_code = True,
                use_fast = True
            )

            self.model = AutoModel.from_pretrained(
                self.model_config['path'],
                trust_remote_code = True,
                dtype = torch.bfloat16
            ).to(self.device).eval()

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"

    def generate(self, image, text):
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        image.convert("RGB"),
                        text
                    ]
                }
            ]

            self.history = conversation.copy()

            response = self.model.chat(
                msgs = conversation,
                image = image,
                tokenizer = self.processor,
                sampling = True,
                temperature = 0.7
            )

            self.history.append({
                'role': 'assistant',
                'content': [response]
            })

            return response

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
        
    def continue_generate(self, text):
        try:
           
            user_message = {
                "role": "user",
                "content": [
                    text
                ]
            }
        
            self.history.append(user_message)
            
            response = self.model.chat(
                msgs = self.history,
                tokenizer = self.processor,
                sampling = True,
                temperature = 0.7
            )

            self.history.append({
                'role': 'assistant',
                'content': [response]
            })

            return response

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
    
    def unload(self):
        return super().unload()