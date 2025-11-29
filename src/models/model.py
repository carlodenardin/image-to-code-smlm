from abc import *

import gc
import torch

class Model(ABC):
    
    def __init__(self, model_config):
        self.history = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image = None
        self.model_config = model_config
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, image, text):
        pass
    
    def unload(self):
        del self.model
        del self.processor

        self.model, self.processor = None, None

    def reset_conversation(self):
        self.history = []
        self.image = None
        
        if hasattr(self.model, 'reset_kv_cache'):
            self.model.reset_kv_cache()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()