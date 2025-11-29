from ast import literal_eval

import os
import logging
import json

logger = logging.getLogger(__name__)

class TestManager():
    
    def __init__(self, test, test_path):
        self.test = test
        self.test_path = test_path

    def load_test(self, test_type):
        try:
            with open(os.path.join(self.test_path, self.test, test_type), 'r') as f:
                data = json.load(f)[0]
            
            inputs = literal_eval(data['input']) if isinstance(data['input'], str) else data['input']
            outputs = literal_eval(data['output']) if isinstance(data['output'], str) else data['output']

            if self.test == "p126":
                outputs = ["Yes" if o else "No" for o in outputs]
            
            return list(zip(inputs, outputs))
            
        except Exception as e:
            logger.error(str(e))
            return []