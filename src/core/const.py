from models.gemma import Gemma
from models.internvl import InternVL
from models.lfm2vl import LFM2VL
from models.minicpm import MiniCPM
from models.gptmini import GPTMini

MODELS = {
    "GPT 5 Mini": {
        "class": GPTMini,
        "name": "GPT 5 Mini",
        "path": "gpt-5-mini"
    },
}

"""
    "Gemma 3 4B": {
        "class": Gemma,
        "name": "Gemma 3 4B",
        "path": "google/gemma-3-4b-it"
    },
    "Gemma 3 4B 4bit": {
        "class": Gemma,
        "name": "Gemma 3 4B 4bit",
        "path": "unsloth/gemma-3-4b-it-bnb-4bit"
    },
    
    "Intern 3.5 1B": {
        "class": InternVL,
        "name": "Intern 3.5 1B",
        "path": "OpenGVLab/InternVL3_5-1B"
    },
    "Intern 3.5 2B": {
        "class": InternVL,
        "name": "Intern 3.5 2B",
        "path": "OpenGVLab/InternVL3_5-2B"
    },
    "LFM2 VL 450M": {
        "class": LFM2VL,
        "name": "LFM2 VL 450M",
        "path": "LiquidAI/LFM2-VL-450M",
    },
    "LFM2 VL 1.6B": {
        "class": LFM2VL,
        "name": "LFM2 VL 1.6B",
        "path": "LiquidAI/LFM2-VL-1.6B"
    },
    "MiniCPM V 4.0": {
        "class": MiniCPM,
        "name": "MiniCPM V 4.0",
        "path": "openbmb/MiniCPM-V-4"
    },
"""

### PATH ###

DB_PATH = "/home/carlodenardin/HDD/Image-to-Code-SMLM/results/results.db"
DB_PATH_NOVEL = "/home/carlodenardin/HDD/Image-to-Code-SMLM/results/results_novel.db"
OUTPUT_PATH = "/home/carlodenardin/HDD/Image-to-Code-SMLM/pipeline_log"
OUTPUT_PATH_NOVEL = "/home/carlodenardin/HDD/Image-to-Code-SMLM/pipeline_log_novel"
INPUT_PATH = "/home/carlodenardin/HDD/Image-to-Code-SMLM/data"
INPUT_PATH_NOVEL = "/home/carlodenardin/HDD/Image-to-Code-SMLM/data_novel"
TEST_CASE_PATH = "/home/carlodenardin/HDD/Image-to-Code-SMLM/problems"


### PROBLEMS ###
PROBLEMS = ["p084", "p106", "p108", "p119", "p120", "p126", "p131", "p147", "p150", "p155"]
DIAGRAMS = ["fc", "block"]
LEVELS = ["1", "2", "3"]

N_RUN = 1

MAX_REPROMPT = 4

### PROMPT ###
EVALUATION_PROMPT = """Given a diagram of an algorithm, generate syntactically correct Python code that implements the logic shown within functions. Include all the codes in ```python ```"""
REASONING_PROMPT = """Analyze the diagram and describe:\n input: input values\n process: what happens step by step\n output: return values"""
GPT_PROMPT = """Given this process, generate syntactically correct Python code that implements the logic shown within functions. Use arguments an return statement"""

REPROMPT_SYNTAX_ERROR = """Fix the syntax error present in the provided code: """
REPROMPT_INPUT = """Code must accept input through function parameters, not input(). Correct the code"""
REPROMPT_RETURN = """Code must return result using a return statement. Correct the code"""
REPROMPT_RETURN_CONCATENATION = """Code must return only the result without additional string or information. Correct the code"""
REPROMPT_ENTRY_POINT_NOT_FOUND = """The entry point of the function has not be found. Implement the logic using functions"""
REPROMPT_CODE_NOT_FOUND = """The entry point of the function has not be found. Implement the logic using functions"""
REPROMPT_RUNTIME_ERROR = """Fix the runtime errors and rewrite the complete corrected code: {errors}"""
REPROMPT_EXECUTION_TIMEOUT = """Execution timeout. Check if there are infinite loops and correct the code"""