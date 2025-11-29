import ast
import textwrap


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.func_calls = {}
        self.current_func = None
        self.func_issues = {}

    def get_entry_point(self):
        defined_funcs = set(self.func_calls.keys())
        called_funcs = set()
        for called in self.func_calls.values():
            called_funcs.update(called)
        called_defined = called_funcs.intersection(defined_funcs)
        entry_points = defined_funcs - called_defined
        return entry_points

    def visit_FunctionDef(self, node):
        self.current_func = node.name
        self.func_calls.setdefault(self.current_func, set())
        self.func_issues.setdefault(self.current_func, [])

        self.has_input = False
        self.has_return = False

        self.generic_visit(node)

        if self.has_input:
            self.func_issues[self.current_func].append("input-argument")

        if not self.has_return:
            self.func_issues[self.current_func].append("missing-return")

        self.current_func = None

    def visit_Call(self, node):
        if self.current_func is not None:
            func_name = self._get_call_name(node.func)
            if func_name:
                if func_name != self.current_func:
                    self.func_calls[self.current_func].add(func_name)
                if func_name == "input":
                    self.has_input = True
        self.generic_visit(node)

    def _get_call_name(self, func_node):
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        else:
            return None

    def visit_Return(self, node):
        if self.current_func is not None:
            self.has_return = True
            if node.value is not None and self._return_has_string_concat(node.value):
                self.func_issues[self.current_func].append("return-with-string")
        self.generic_visit(node)

    def _return_has_string_concat(self, node):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                if self._contains_str(node.left) or self._contains_str(node.right):
                    return True
                return (
                    self._return_has_string_concat(node.left)
                    or self._return_has_string_concat(node.right)
                )
            return False
        elif isinstance(node, ast.JoinedStr):
            return True
        return False

    def _contains_str(self, node):
        return isinstance(node, ast.Constant) and isinstance(node.value, str)


class CodeAnalyzerManager:
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.issues = {
            "syntax_errors": [],
            "function_issues": {},
            "entry_point": [],
        }

    def run_syntax_check(self):
        try:
            tree = ast.parse(textwrap.dedent(self.code))
            func_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            return ast.Module(body=func_defs, type_ignores=[])
        except SyntaxError as e:
            self.issues["syntax_errors"].append(f"Syntax error: {e}")
            return None

    def run_analysis(self, tree):
        if tree is not None:
            self.analyzer.visit(tree)

    def check_issues(self):
        if self.analyzer.func_issues:
            self.issues["function_issues"] = self.analyzer.func_issues
        try:
            entry = self.analyzer.get_entry_point()
            if entry:
                self.issues["entry_point"] = list(entry)
            else:
                self.issues["entry_point"] = []
        except Exception as e:
            self.issues["entry_point"] = []
            self.issues["syntax_errors"].append(str(e))

    def run(self, code):
        self.code = code
        tree = self.run_syntax_check()
        self.run_analysis(tree)
        self.check_issues()
        return self.issues


if __name__ == "__main__":
    code = '''
def generate_list(n):
    if n == 0:
        return []
    elif n == 1:
        return [0]
    elif n > 1:
        if n % 2 == 0:
            return generate_list(n // 2)
        else:
            return "Result: " + [0] + generate_list(n - 1)
    else:
        return [0] + generate_list(n - 1)
'''

    pipeline = CodeAnalyzerManager()
    results = pipeline.run(code)
    print("Syntax errors:", results["syntax_errors"])
    print("Function problems:", results["function_issues"])
    print("Entry point:", results["entry_point"])
