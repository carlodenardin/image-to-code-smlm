from utils.const import *

from PIL import Image

import os

def extract_dataset():
    filtered_diagrams = {}
    for diagram in DIAGRAMS:
        if diagram == "block":
            filtered_diagrams[diagram] = [lvl for lvl in LEVELS if lvl == "1"]
        else:
            filtered_diagrams[diagram] = LEVELS

    return [
        (problem, diagram, level)
        for problem in PROBLEMS
        for diagram, level_list in filtered_diagrams.items()
        for level in level_list
    ]
    
def preprocess_image(image, target_size=448):
    image = image.convert('RGB')
    image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    offset_x = (target_size - image.width) // 2
    offset_y = (target_size - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))

    return canvas

def save_data(data, file_path, file_name):
    os.makedirs(file_path, exist_ok = True)

    with open(os.path.join(file_path, file_name), 'w', encoding='utf-8') as f:
        f.write(data)

def get_reprompt(issues):
        reprompt = ""
        # Check Syntax Errors
        if issues['syntax_errors']:
            reprompt += REPROMPT_SYNTAX_ERROR + issues['syntax_errors'][0] + "\n"
        else:
            # Check Function Issues (if syntax error is present no function issues will be present)
            for f, issues in issues['function_issues'].items():
                for issue in issues:
                    if issue == "input-argument":
                        reprompt += f"Function '{f}': " + REPROMPT_INPUT + "\n"
                    if issue == "missing-return":
                        reprompt += f"Function '{f}': " + REPROMPT_RETURN + "\n"
                    if issue == "return-with-string":
                        reprompt += f"Function '{f}': " + REPROMPT_RETURN_CONCATENATION + "\n"
        
        if reprompt != "":
            reprompt = "Solve this issues in the provided code:\n" + reprompt

        return reprompt