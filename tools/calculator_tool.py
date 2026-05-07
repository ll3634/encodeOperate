#!/usr/bin/env python3
"""
Calculator tool with safe expression evaluation.
"""

import ast
import operator
import math
from typing import Union


# Allowed operators for safe evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed functions
SAFE_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'sqrt': math.sqrt,
    'log': math.log,
    'log10': math.log10,
    'exp': math.exp,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'pi': math.pi,
    'e': math.e,
}


class SafeEvaluator(ast.NodeVisitor):
    """
    Safe expression evaluator using AST.
    Only allows arithmetic operations and safe functions.
    """
    
    def visit_Expression(self, node):
        return self.visit(node.body)
    
    def visit_Num(self, node):
        # Python 3.7 and earlier
        return node.n
    
    def visit_Constant(self, node):
        # Python 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](left, right)
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return SAFE_OPERATORS[op_type](operand)
    
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        
        args = [self.visit(arg) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)
    
    def visit_Name(self, node):
        name = node.id
        if name in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[name]
        raise ValueError(f"Unknown variable: {name}")
    
    def visit_List(self, node):
        return [self.visit(el) for el in node.elts]
    
    def visit_Tuple(self, node):
        return tuple(self.visit(el) for el in node.elts)
    
    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str) -> Union[int, float]:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression string
        
    Returns:
        Numeric result
        
    Raises:
        ValueError: If expression contains unsafe operations
    """
    # Clean up expression
    expression = expression.strip()
    
    # Handle common formatting issues
    expression = expression.replace('^', '**')  # Power notation
    expression = expression.replace('×', '*')   # Multiplication
    expression = expression.replace('÷', '/')   # Division
    
    try:
        tree = ast.parse(expression, mode='eval')
        evaluator = SafeEvaluator()
        result = evaluator.visit(tree)
        return result
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}")


class CalculatorTool:
    """
    Calculator tool for ReAct agent.
    Safely evaluates arithmetic expressions.
    """
    
    def __init__(self, precision: int = 6):
        self.precision = precision
    
    def __call__(self, expression: str) -> str:
        """Evaluate expression and return result as string."""
        try:
            result = safe_eval(expression)
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                return f"{result:.{self.precision}g}"
            return str(result)
            
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Self-test
    calc = CalculatorTool()
    
    tests = [
        ("2 + 3", "5"),
        ("10 * 5", "50"),
        ("100 / 4", "25"),
        ("2 ** 10", "1024"),
        ("sqrt(16)", "4"),
        ("abs(-5)", "5"),
        ("round(3.7)", "4"),
    ]
    
    print("Calculator self-test:")
    for expr, expected in tests:
        result = calc(expr)
        status = "✓" if result == expected else f"✗ (got {result})"
        print(f"  {expr} = {expected} {status}")

