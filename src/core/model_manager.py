import logging

logger = logging.getLogger(__name__)

class ModelManager:

    def __init__(self, model_config):
        self.model_config = model_config
        self.current_model = None

    def load_model(self):
        logger.info(f"Loading model: {self.model_config['name']}")

        try:
            mclass = self.model_config['class']
            self.current_model = mclass(self.model_config)
            self.current_model.load()

            logger.info(f"Model loaded: {self.model_config['name']}")

            return self.current_model

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
        
    def unload_model(self):
        logger.info(f"Unloading model: {self.model_config['name']}")

        try:
            self.current_model.unload()
            self.current_model = None

            logger.info(f"Model unloaded: {self.model_config['name']}")

        except Exception as e:
            logger.error(str(e))
            return f"Error {str(e)}"
    
    def reset_conversation(self):
        self.current_model.reset_conversation() 
        