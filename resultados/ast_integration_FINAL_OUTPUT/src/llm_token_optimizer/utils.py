import ast
import json

def parse_code_to_ast(code: str) -> ast.AST:
    """
    Parses a given code string into an Abstract Syntax Tree (AST).

    Args:
        code: The source code string to parse.

    Returns:
        An AST object representing the parsed code.
    """
    return ast.parse(code)

def unparse_ast_to_code(node: ast.AST) -> str:
    """
    Unparses an AST object back into a code string.

    Args:
        node: The AST node to unparse.

    Returns:
        A string representing the unparsed code.
    """
    return ast.unparse(node)

def serialize_reduction_map(reduction_map: dict) -> str:
    """
    Serializes a reduction map (dictionary) into a JSON string.

    Args:
        reduction_map: A dictionary containing the reduction mapping information.

    Returns:
        A JSON string representation of the reduction map.
    """
    return json.dumps(reduction_map, indent=2)

def deserialize_reduction_map(json_string: str) -> dict:
    """
    Deserializes a JSON string back into a reduction map dictionary.

    Args:
        json_string: The JSON string to deserialize.

    Returns:
        A dictionary representing the deserialized reduction map.
    """
    return json.loads(json_string)