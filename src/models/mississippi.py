from models.model import Model
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
import tempfile
import os
import shutil
import gc


class Mississippi(Model):
    
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.conversation_history = None
        self.last_image_path = None
        self.temp_dir = None
        self.device = 'cuda'

    def load(self):
        try:
            config = AutoConfig.from_pretrained(
                self.model_config['path'],
                trust_remote_code=True
            )

            if hasattr(config, 'llm_config'):
                config.llm_config._attn_implementation = 'eager'

            self.model = AutoModel.from_pretrained(
                self.model_config['path'],
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                device_map=self.device,
                trust_remote_code=True
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['path'],
                trust_remote_code=True,
                use_fast=False
            )

            self.generation_config = {
                'max_new_tokens': 512,
                'do_sample': True,
                'temperature': 0.7
            }

            self.temp_dir = tempfile.mkdtemp()

            return f"Loaded {self.model_config.get('name', 'Mississippi')}."

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def generate(self, image, text, reset=True):
        try:
            if reset:
                self.reset_conversation()

            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()

            temp_image_path = os.path.join(self.temp_dir, "temp_image.png")
            image.convert("RGB").save(temp_image_path)
            self.last_image_path = temp_image_path

            question = f"<image>\n{text}"

            response, history = self.model.chat(
                self.tokenizer,
                temp_image_path,
                question,
                self.generation_config,
                history=self.conversation_history,
                return_history=True
            )

            self.conversation_history = history
            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"

    def continue_generate(self, text):
        try:
            if self.last_image_path is None:
                return "Error: No previous image. Call generate() first."

            response, history = self.model.chat(
                self.tokenizer,
                None,
                text,
                self.generation_config,
                history=self.conversation_history,
                return_history=True
            )

            self.conversation_history = history
            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating follow-up: {str(e)}"

    def reset_conversation(self):
        self.conversation_history = None

        if hasattr(self, 'last_image_path') and self.last_image_path:
            if os.path.exists(self.last_image_path):
                try:
                    os.remove(self.last_image_path)
                except Exception:
                    pass

        self.last_image_path = None

        if hasattr(self, 'model') and self.model and hasattr(self.model, 'reset_kv_cache'):
            self.model.reset_kv_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def unload(self):
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        return super().unload()
