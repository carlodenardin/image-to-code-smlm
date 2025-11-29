from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, TimeoutError
from utils.const import *
import logging
import re
import os
import subprocess
import json
import time

logger = logging.getLogger(__name__)


def _execute_single_test_with_subprocess_timeout(code, entry_point, test_case, timeout):
    input_data, expected_output = test_case
    
    script = f"""
import json
import sys

{code}

try:
    input_data = {repr(input_data)}
    expected_output = {repr(expected_output)}
    
    if isinstance(input_data, tuple):
        result = {entry_point}(*input_data)
    else:
        result = {entry_point}(input_data)
    
    if isinstance(result, bool):
        result = "Yes" if result else "No"
    
    passed = result == expected_output
    
    print(json.dumps({{
        'input': input_data,
        'expected': expected_output,
        'actual': result,
        'passed': passed,
        'error': None
    }}))
except Exception as e:
    print(json.dumps({{
        'input': {repr(input_data)},
        'expected': {repr(expected_output)},
        'actual': None,
        'passed': False,
        'error': str(e)
    }}))
"""
    
    try:
        result = subprocess.run(
            ['python3', '-c', script],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {
                'input': input_data,
                'expected': expected_output,
                'actual': None,
                'passed': False,
                'error': f'Invalid JSON output: {result.stdout[:200]}'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'input': input_data,
            'expected': expected_output,
            'actual': None,
            'passed': False,
            'error': REPROMPT_EXECUTION_TIMEOUT
        }
    except Exception as e:
        return {
            'input': input_data,
            'expected': expected_output,
            'actual': None,
            'passed': False,
            'error': str(e)
        }


class EvaluationManager():
    def __init__(self, code, entry_point, timeout = 0.4):
        self.code = code
        self.entry_point = entry_point
        self.timeout = timeout

    def generalize_error(self, error_msg):
        if not error_msg:
            return error_msg
        
        if "cannot access local variable" in error_msg:
            return "cannot access local variable (naming conflict)"
        
        if "Execution timeout" in error_msg or "Timed out" in error_msg:
            return REPROMPT_EXECUTION_TIMEOUT
        
        patterns = [
            (r":\s*['\"].*?['\"]", ":"),
            (r":\s*\d+", ":"),
            (r"\(.*?\)", "()"),
        ]
        
        generalized = error_msg
        for pattern, replacement in patterns:
            generalized = re.sub(pattern, replacement, generalized)
        
        return generalized.strip().rstrip(':').strip()

    def run_tests(self, tests, num_workers = None):
        
        if num_workers is None:
            num_workers = max(1, min(16, os.cpu_count(), len(tests)))

        logging.info(f"Evaluating {len(tests)} cases using {num_workers} cores")

        results = []
        errors = set()
        max_total_time = (self.timeout + 2) * len(tests) + 20
        start_time = time.time()

        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                
                future_to_test = {
                    executor.submit(
                        _execute_single_test_with_subprocess_timeout, 
                        self.code,
                        self.entry_point,
                        test, 
                        self.timeout
                    ): (idx, test)
                    for idx, test in enumerate(tests)
                }
                
                pending = set(future_to_test.keys())
                completed_count = 0
                
                while pending:
                    elapsed = time.time() - start_time
                    
                    if elapsed > max_total_time:
                        break
                    
                    remaining_time = min(self.timeout + 2, max_total_time - elapsed)
                    done, pending = wait(pending, timeout = remaining_time, return_when = FIRST_COMPLETED)
                    
                    if not done and pending:
                        continue
                    
                    for future in done:
                        completed_count += 1
                        idx, test_case = future_to_test[future]
                        
                        try:
                            result = future.result(timeout=0.5)
                            results.append((idx, result))
                            
                        except TimeoutError:
                            results.append((idx, {
                                'input': test_case[0],
                                'expected': test_case[1],
                                'actual': None,
                                'passed': False,
                                'error': REPROMPT_EXECUTION_TIMEOUT
                            }))
                            
                        except Exception as e:
                            results.append((idx, {
                                'input': test_case[0],
                                'expected': test_case[1],
                                'actual': None,
                                'passed': False,
                                'error': str(e)
                            }))
                
                if pending:
                    for future in pending:
                        future.cancel()
                        idx, test_case = future_to_test[future]
                        results.append((idx, {
                            'input': test_case[0],
                            'expected': test_case[1],
                            'actual': None,
                            'passed': False,
                            'error': REPROMPT_EXECUTION_TIMEOUT
                        }))
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            if not results:
                for idx, test in enumerate(tests):
                    results.append((idx, {
                        'input': test[0],
                        'expected': test[1],
                        'actual': None,
                        'passed': False,
                        'error': f'Executor failed: {str(e)}'
                    }))
        
        results.sort(key=lambda x: x[0])
        results = [r[1] for r in results]
        
        logging.info(f"All test processed: {len(results)} results, {len([r for r in results if r['passed']])} passed")
        
        for result in results:
            error = result.get('error')
            if error and error != 'None':
                generalized_error = self.generalize_error(error)
                errors.add(generalized_error)
        
        if errors:
            logging.info(f"Found {len(errors)} unique error types")

        return results, errors

if __name__ == '__main__':

    CODE = """
def get_input(value):
    if isinstance(value, int):
        n = value
    else:
        try:
            n = int(str(value).strip())
        except Exception:
            raise ValueError("Invalid integer input")
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    return n
def get_total_sum_of_digits(n):
    return sum(int(d) for d in str(n))
def return_number_of_digits_in_binary(value):
    if value == 0:
        return 1
    return value.bit_length()
def main(raw_value):
    n = get_input(raw_value)
    total = get_total_sum_of_digits(n)
    return return_number_of_digits_in_binary(total)
"""

    em = EvaluationManager(CODE, 'main')

    result = em.run_tests(([1000], [1]))

    print(result)