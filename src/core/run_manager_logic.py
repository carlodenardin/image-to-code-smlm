from core.answer_analyzer import AnswerAnalyzerManager
from core.code_analyzer import CodeAnalyzerManager
from core.database_manager import DatabaseManager
from core.evaluation_manager import EvaluationManager
from core.model_manager import ModelManager
from core.test_manager import TestManager
from utils.const import *
from utils.utils import *

from PIL import Image

import logging
import os
import time
import torch

class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",     # blu
        "INFO": "\033[92m",      # verde
        "WARNING": "\033[93m",   # giallo
        "ERROR": "\033[91m",     # rosso
        "CRITICAL": "\033[95m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

def setup_colored_logging():
    handler = logging.StreamHandler()
    formatter = ColorFormatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[handler])

setup_colored_logging()
logger = logging.getLogger(__name__)

class RunMangerLogic():
    def __init__(self, model_config):
        self.database_manager = DatabaseManager(DB_PATH_LOGIC)
        self.dataset = extract_dataset()
        self.model_config = model_config
        self.model_manager = ModelManager(model_config)
        self.model = None

    def save_step(self, content, path, prefix, nreprompt, counter):
        save_data(content, path, f"{counter:02d}_{prefix}_{nreprompt}.txt")
        return counter + 1

    def run_tests_and_collect_errors(self, code, entry_point, problem_info, model_name, run_name, reprompt_info):
        test_loader = TestManager(problem_info, TEST_CASE_PATH)
        generated_tests = test_loader.load_test("generated.jsonl")
        official_tests = test_loader.load_test("official.jsonl")

        run = EvaluationManager(code, entry_point)
        results_official, errors_official = run.run_tests(official_tests)
        self.database_manager.insert_results(
            results_official,
            model_name,
            run_name,
            reprompt_info,
            problem_info[0],
            problem_info[1],
            problem_info[2],
            "official",
        )

        results_generated, errors_generated = run.run_tests(generated_tests)
        self.database_manager.insert_results(
            results_generated,
            model_name,
            run_name,
            reprompt_info,
            problem_info[0],
            problem_info[1],
            problem_info[2],
            "generated",
        )

        return errors_official.union(errors_generated)

    def create_static_error_results(self, tests, static_error):
        results = []
        for test in tests:
            input_data, expected_output = test
            results.append({
                "input": input_data,
                "expected": expected_output,
                "actual": None,
                "passed": False,
                "error": f"Static error: {static_error}",
            })
        return results

    def save_static_error_to_db(self, problem_info, model_name, run_name, reprompt_info, static_error):
        test_loader = TestManager(problem_info, TEST_CASE_PATH)
        generated_tests = test_loader.load_test("generated.jsonl")
        official_tests = test_loader.load_test("official.jsonl")

        results_official = self.create_static_error_results(official_tests, static_error)
        results_generated = self.create_static_error_results(generated_tests, static_error)

        self.database_manager.insert_results(
            results_official,
            model_name,
            run_name,
            reprompt_info,
            problem_info[0],
            problem_info[1],
            problem_info[2],
            "official",
        )
        self.database_manager.insert_results(
            results_generated,
            model_name,
            run_name,
            reprompt_info,
            problem_info[0],
            problem_info[1],
            problem_info[2],
            "generated",
        )

    def run_evaluation_pipeline(self, problem, model_name, run_name):
        p, d, l = problem

        pipeline_log_path = os.path.join(
            OUTPUT_PATH_LOGIC, model_name, run_name, p, d, f"l{l}"
        )
        logic_path = os.path.join(INPUT_PATH_LOGIC, model_name, run_name, p, d, f"l{l}", "00_reasoning.txt")

        with open(logic_path, "r", encoding="utf-8") as f:
            logic = f.read()

        self.model.reset_conversation()
        logging.info(f"Evaluating {model_name}/{run_name}/{p}/{d}/l{l} using {self.model_config['name']}")

        response = self.model.generate_text(logic, GPT_PROMPT)
        fcounter = 0
        
        for i in range(MAX_REPROMPT + 1):
            nreprompt = i + 1
            logging.info(f"Prompt N° {nreprompt}/{MAX_REPROMPT + 1}")
            fcounter = self.save_step(response, pipeline_log_path, "response", nreprompt, fcounter)

            logging.info("Analyzing response...")
            answer_analyzer = AnswerAnalyzerManager()
            code = answer_analyzer.run(response)
            fcounter = self.save_step(code, pipeline_log_path, "code", nreprompt, fcounter)

            if code != "code-not-found":
                logging.info("Code found in response. Analyzing the code")
                code_analyzer = CodeAnalyzerManager()
                analysis = code_analyzer.run(code)
                fcounter = self.save_step(str(analysis), pipeline_log_path, "analysis", nreprompt, fcounter)

                reprompt = get_reprompt(analysis)

                if reprompt:
                    logging.warning("Static errors found in the code.")
                    self.save_static_error_to_db(problem, model_name, run_name, f"reprompt_{nreprompt}", reprompt)

                    if i < MAX_REPROMPT:
                        logging.info("Sending reprompt")
                        fcounter = self.save_step(reprompt, pipeline_log_path, "reprompt", nreprompt, fcounter)
                        response = self.model.continue_generate_text(reprompt)
                    else:
                        logging.warning("Max reprompt reached")
                        fcounter = self.save_step(reprompt, pipeline_log_path, "reprompt_FINAL", nreprompt, fcounter)
                        break

                elif analysis.get("entry_point"):
                    logging.info("Static errors not found")
                    entry_point = analysis["entry_point"][0]

                    if entry_point:
                        logging.info(f"Entry point found: {entry_point}")
                        runtime_errors = self.run_tests_and_collect_errors(
                            code, entry_point, problem, model_name, run_name, f"reprompt_{nreprompt}"
                        )

                        if runtime_errors:
                            logging.error(f"Runtime errors found: {len(runtime_errors)}")
                            errors_str = "\n".join(f"  - {err}" for err in runtime_errors)
                            reprompt = REPROMPT_RUNTIME_ERROR.format(errors=errors_str)

                            if i < MAX_REPROMPT:
                                logging.info("Reprompting due to runtime errors")
                                fcounter = self.save_step(reprompt, pipeline_log_path, "reprompt_runtime", nreprompt, fcounter)
                                response = self.model.continue_generate_text(reprompt)
                            else:
                                logging.warning("Max reprompting reached")
                                break
                        else:
                            logging.info("Test completed successfully")
                            break
                    else:
                        logging.warning("Entry point not found")
                        if i < MAX_REPROMPT:
                            logging.info("Reprompting: entry point not found")
                            self.save_static_error_to_db(problem, model_name, run_name, f"reprompt_{nreprompt}", REPROMPT_ENTRY_POINT_NOT_FOUND)
                            fcounter = self.save_step(REPROMPT_ENTRY_POINT_NOT_FOUND, pipeline_log_path, "reprompt_entry_not_found", nreprompt, fcounter)
                            response = self.model.continue_generate_text(REPROMPT_ENTRY_POINT_NOT_FOUND)
                        else:
                            logging.warning("Max reprompting reached")
                            break
            else:
                logging.warning("Code implementation not found")
                if i < MAX_REPROMPT:
                    logging.info("Reprompting: code not found")
                    self.save_static_error_to_db(problem, model_name, run_name, f"reprompt_{nreprompt}", REPROMPT_CODE_NOT_FOUND)
                    fcounter = self.save_step(REPROMPT_CODE_NOT_FOUND, pipeline_log_path, "reprompt_code_not_found", nreprompt, fcounter)
                    response = self.model.continue_generate(REPROMPT_CODE_NOT_FOUND)
                else:
                    logging.warning("Max reprompting reached")
                    break

    def run_pipeline(self):
        for model_name in MODELS_NAME:
            for i in range(N_RUN):
                logger.info(f"Run N° {i + 1}")
                for problem in self.dataset:
                    try:
                        self.run_evaluation_pipeline(problem, model_name, f"run_{i + 1}")
                    except Exception as e:
                        logger.error(str(e))
                        return f"Error {str(e)}"

    def run(self):
        self.model = self.model_manager.load_model()
        self.run_pipeline()
