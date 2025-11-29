from core.run_manager_novel import RunMangerNovel
from utils.const import MODELS

import logging
import os
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"
multiprocessing.set_start_method('fork', force = True)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    logging.info("Initialize evaluation pipeline")

    for model_name, model_config in MODELS.items():
        logging.info(f"Initialize model {model_name}")
        run_manager = RunMangerNovel(model_config)
        run_manager.run()
