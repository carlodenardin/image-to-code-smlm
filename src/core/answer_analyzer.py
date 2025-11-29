import re

class AnswerAnalyzer():
    
    def extract_code(self, text):
        pattern = r'```python\s*\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            code_blocks = [self._clean_code(block.strip()) for block in matches if block.strip()]
            return '\n\n'.join(code_blocks)
        
        pattern_generic = r'```(.*?)```'
        matches_generic = re.findall(pattern_generic, text, re.DOTALL)
        
        code_blocks_generic = []
        for block in matches_generic:
            if self._looks_like_python(block.strip()):
                code_blocks_generic.append(self._clean_code(block.strip()))
        
        if code_blocks_generic:
            return '\n\n'.join(code_blocks_generic)
        
        code_block = self._extract_largest_code_block(text)
        if code_block:
            return self._clean_code(code_block)
        
        return "code-not-found"

    def _clean_code(self, code: str) -> str:
        code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
        code = re.sub(r'#.*', '', code)

        lines = code.split('\n')
        cleaned_lines = []
        inside_main = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("if __name__") and "__main__" in stripped:
                inside_main = True
                continue

            if inside_main:
                if stripped and not line.startswith(' '):
                    inside_main = False
                else:
                    continue

            if stripped.startswith("print(") or stripped.startswith("print "):
                indent = len(line) - len(line.lstrip())
                cleaned_lines.append(" " * indent + "pass")
                continue

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()
    
    def _looks_like_python(self, text):
        python_indicators = [
            r'\bdef\s+\w+\s*\(',
            r'\bclass\s+\w+',
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import',
            r'\breturn\s+',
            r'\bif\s+.*:',
            r'\bfor\s+\w+\s+in\s+',
            r'\bwhile\s+.*:',
        ]
        
        for pattern in python_indicators:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_largest_code_block(self, text):
        lines = text.split('\n')
        in_code_block = False
        current_block = []
        all_blocks = []
        
        for line in lines:
            stripped = line.strip()
            
            is_start_of_code = re.match(r'^(def|class|import|from)\s+\w+', stripped)
            
            if is_start_of_code:
                if not in_code_block and not current_block:
                    in_code_block = True
                elif in_code_block and not line[0].isspace() and stripped not in current_block[-1]: 
                    pass 

                current_block.append(line)

            elif in_code_block:
                if line and (line[0].isspace() or not stripped or stripped.startswith('#')):
                    current_block.append(line)
                
                elif stripped and not line[0].isspace():
                    all_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
        
        if current_block:
            all_blocks.append('\n'.join(current_block))
        
        if all_blocks:
            return max(all_blocks, key=len)
        
        return None


class AnswerAnalyzerManager():
    def __init__(self):
        self.analyzer = AnswerAnalyzer()
    
    def run(self, answer):
        clean_code = self.analyzer.extract_code(answer)
        return clean_code