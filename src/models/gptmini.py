from models.model import Model
from openai import OpenAI

from dotenv import load_dotenv
from io import BytesIO

import base64
import logging
import os
from PIL import Image

logger = logging.getLogger(__name__)
load_dotenv()

class GPTMini(Model):

    def __init__(self, model_config):
        super().__init__(model_config)
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.last_image_b64 = None 
        self.history = [] 

    def load(self):
        try:
            logger.info(f"Using OpenAI model: {self.model_config['name']}")
        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"

    def _generate_helper(self):
        try:
            response_obj = self.client.chat.completions.create(
                model = self.model_config['path'],
                messages = self.history,
                max_completion_tokens = 2048, 
            )

            if (response_obj.choices and 
                response_obj.choices[0].message and 
                response_obj.choices[0].message.content is not None):
                
                response_text = response_obj.choices[0].message.content

                self.history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_text}]
                })
                return response_text
            else:
                logger.warning("Risposta del modello vuota o senza contenuto.")
                return ""

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"

    def generate(self, image: Image.Image, text: str):
        try:
            image_b64 = self._image_to_base64(image)
            self.last_image_b64 = image_b64 

            user_message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": text}
                ]
            }

            self.history = [user_message]

            return self._generate_helper()

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"

    def continue_generate(self, text: str):
        try:
            if not self.last_image_b64:
                return "Error: Cannot continue without initial image context."

            user_message = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.last_image_b64}}, 
                    {"type": "text", "text": text}
                ]
            }
            
            self.history.append(user_message)

            return self._generate_helper()

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"

    def generate_text(self, text: str, prompt):
        try:
            self.last_image_b64 = None
            self.history = [{
                "role": "user",
                "content": [{"type": "text", "text": f"{prompt} \n {text}"}]
            }]
            return self._generate_helper()
        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"

    def continue_generate_text(self, prompt):
        try:
            if not self.history:
                return "Error: No previous conversation. Use generate_text() first."
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": f"{prompt}"}]
            }
            self.history.append(user_message)
            return self._generate_helper()
        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"

    def unload(self):
        self.history = []
        self.last_image_b64 = None
        return super().unload()
