import pytest
from src.llm_token_optimizer.reducer import reduce_code

# Test cases for basic reduction: comments and whitespace
def test_remove_single_line_comment():
    code = "x = 1 # This is a comment"
    expected_reduced_code = "x = 1"
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    assert len(reduction_map) == 1
    assert reduction_map[0]["original_line_num"] == 0
    assert reduction_map[0]["original_content"] == code
    assert reduction_map[0]["reduced_content"] == expected_reduced_code
    assert reduction_map[0]["type"] == "kept"

def test_remove_multiple_standalone_comments():
    code = """
# Comment 1
x = 1
# Comment 2
y = 2 # Inline comment
# Comment 3
"""
    expected_reduced_code = "x = 1\ny = 2"
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    assert len(reduction_map) == 7 # 7 original lines
    assert reduction_map[0]["type"] == "removed_empty"
    assert reduction_map[1]["type"] == "removed_comment"
    assert reduction_map[2]["reduced_content"] == "x = 1"
    assert reduction_map[3]["type"] == "removed_comment"
    assert reduction_map[4]["reduced_content"] == "y = 2"
    assert reduction_map[5]["type"] == "removed_comment"
    assert reduction_map[6]["type"] == "removed_empty"


def test_remove_excess_whitespace_and_empty_lines():
    code = """
    def func():

        pass


    """
    expected_reduced_code = "def func():\n    pass"
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    assert len(reduction_map) == 7 # Original lines
    assert reduction_map[0]["type"] == "removed_empty"
    assert reduction_map[1]["reduced_content"] == "def func():"
    assert reduction_map[2]["type"] == "removed_empty"
    assert reduction_map[3]["reduced_content"] == "    pass" # Indentation should be preserved
    assert reduction_map[4]["type"] == "removed_empty"
    assert reduction_map[5]["type"] == "removed_empty"
    assert reduction_map[6]["type"] == "removed_empty"


# Test cases for edge cases
def test_empty_code():
    code = ""
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == ""
    assert reduction_map == []

def test_code_with_only_whitespace():
    code = "   \n\t\n     "
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == ""
    assert all(item["type"].startswith("removed") for item in reduction_map)
    assert len(reduction_map) == 3

def test_code_with_only_comments():
    code = "# Comment 1\n# Comment 2"
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == ""
    assert all(item["type"] == "removed_comment" for item in reduction_map)
    assert len(reduction_map) == 2

def test_single_line_code_no_comments():
    code = "print('Hello')"
    expected_reduced_code = "print('Hello')"
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    assert len(reduction_map) == 1
    assert reduction_map[0]["type"] == "kept"

# Test cases for complex strings
def test_code_with_docstrings():
    code = """
def my_function(a, b):
    \"\"\"
    This is a docstring.
    It should be preserved.
    \"\"\"
    result = a + b
    return result
"""
    # Docstrings should be preserved entirely, including their internal structure and indentation.
    # Only surrounding empty lines or comments should be removed.
    expected_reduced_code = """def my_function(a, b):
    \"\"\"
    This is a docstring.
    It should be preserved.
    \"\"\"
    result = a + b
    return result"""
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    assert len(reduction_map) == 8 # Original lines count
    # Check that docstring lines are marked as kept and content is preserved
    assert reduction_map[2]["reduced_content"] == '    """'
    assert reduction_map[3]["reduced_content"] == '    This is a docstring.'
    assert reduction_map[4]["reduced_content"] == '    It should be preserved.'
    assert reduction_map[5]["reduced_content"] == '    """'


def test_code_with_docstrings_and_inline_comments():
    code = """
def another_func():
    \"\"\"Docstring here.\"\"\" # Inline comment next to docstring
    # Another standalone comment line
    pass # Inline comment
"""
    expected_reduced_code = """def another_func():
    \"\"\"Docstring here.\"\"\"
    pass"""
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    assert len(reduction_map) == 6
    assert reduction_map[2]["reduced_content"] == '    """Docstring here."""' # Inline comment removed
    assert reduction_map[3]["type"] == "removed_comment" # Standalone comment removed
    assert reduction_map[4]["reduced_content"] == "    pass" # Inline comment removed

def test_code_with_various_string_types():
    code = r"""
message = "Hello, world!" # Standard string
path = r'C:\Users\Name\file.txt' # Raw string
query = f"SELECT * FROM users WHERE name='{user_name}'" # F-string
long_text = '''Line 1
Line 2 with #hash and "quotes"
Line 3''' # Multi-line string
"""
    expected_reduced_code = r"""message = "Hello, world!"
path = r'C:\Users\Name\file.txt'
query = f"SELECT * FROM users WHERE name='{user_name}'"
long_text = '''Line 1
Line 2 with #hash and "quotes"
Line 3'''"""
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    assert len(reduction_map) == 8
    assert reduction_map[1]["reduced_content"] == 'message = "Hello, world!"'
    assert reduction_map[2]["reduced_content"] == r"path = r'C:\Users\Name\file.txt'"
    assert reduction_map[3]["reduced_content"] == r"query = f\"SELECT * FROM users WHERE name='{user_name}'\""
    assert reduction_map[4]["reduced_content"] == "long_text = '''Line 1"
    assert reduction_map[5]["reduced_content"] == 'Line 2 with #hash and "quotes"'
    assert reduction_map[6]["reduced_content"] == "Line 3'''"
    assert all(item["type"] == "kept" for item in [reduction_map[1], reduction_map[2], reduction_map[3], reduction_map[4], reduction_map[5], reduction_map[6]])


# Test cases for reduction map integrity
def test_reduction_map_integrity_detailed():
    code = """
import os # Import statement
def process_data(data):
    # Check if data is valid
    if data is None:
        return None # Early exit

    # Process each item
    for item in data:
        # Some complex logic
        if item > 0:
            result = item * 2
        else:
            result = item / 2 # Division
        print(f"Processed: {result}") # Debugging line
    return data # Return original data
"""
    reduced_code, reduction_map = reduce_code(code)

    # Verify map length matches original lines
    original_lines = code.splitlines()
    assert len(reduction_map) == len(original_lines)

    # Verify types and content for specific lines
    # Line 0: empty
    assert reduction_map[0]["original_line_num"] == 0
    assert reduction_map[0]["original_content"] == ""
    assert reduction_map[0]["reduced_content"] == ""
    assert reduction_map[0]["type"] == "removed_empty"

    # Line 1: import os # Import statement
    assert reduction_map[1]["original_line_num"] == 1
    assert reduction_map[1]["original_content"] == "import os # Import statement"
    assert reduction_map[1]["reduced_content"] == "import os"
    assert reduction_map[1]["type"] == "kept"

    # Line 3: # Check if data is valid
    assert reduction_map[3]["original_line_num"] == 3
    assert reduction_map[3]["original_content"] == "    # Check if data is valid"
    assert reduction_map[3]["reduced_content"] == ""
    assert reduction_map[3]["type"] == "removed_comment"

    # Line 5: return None # Early exit
    assert reduction_map[5]["original_line_num"] == 5
    assert reduction_map[5]["original_content"] == "        return None # Early exit"
    assert reduction_map[5]["reduced_content"] == "        return None"
    assert reduction_map[5]["type"] == "kept"

    # Line 7: # Process each item
    assert reduction_map[7]["original_line_num"] == 7
    assert reduction_map[7]["original_content"] == "    # Process each item"
    assert reduction_map[7]["reduced_content"] == ""
    assert reduction_map[7]["type"] == "removed_comment"

    # Line 13: result = item / 2 # Division
    assert reduction_map[13]["original_line_num"] == 13
    assert reduction_map[13]["original_content"] == "            result = item / 2 # Division"
    assert reduction_map[13]["reduced_content"] == "            result = item / 2"
    assert reduction_map[13]["type"] == "kept"

    # Line 14: print(f"Processed: {result}") # Debugging line
    assert reduction_map[14]["original_line_num"] == 14
    assert reduction_map[14]["original_content"] == "        print(f\"Processed: {result}\") # Debugging line"
    assert reduction_map[14]["reduced_content"] == "        print(f\"Processed: {result}\")"
    assert reduction_map[14]["type"] == "kept"

    # Ensure no unexpected types or missing info
    for item in reduction_map:
        assert "original_line_num" in item
        assert "original_content" in item
        assert "reduced_content" in item
        assert "type" in item
        assert item["type"] in ["kept", "removed_comment", "removed_whitespace", "removed_empty"]


def test_no_change_for_already_optimized_code():
    code = """def func():
    return 1"""
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == code
    assert len(reduction_map) == 2
    assert reduction_map[0]["reduced_content"] == "def func():"
    assert reduction_map[1]["reduced_content"] == "    return 1"
    assert all(item["type"] == "kept" for item in reduction_map)

# Additional test for complex scenarios:
def test_code_with_mixed_content_and_indentation():
    code = """
class MyClass:
    def __init__(self):
        # Initialize something
        self.value = 0 # Default value

    def calculate(self, x):
        '''
        Calculates a value based on x.
        '''
        if x > 10: # Condition
            return x * self.value
        else:
            return x + self.value # Addition operation
        # End of method
"""
    expected_reduced_code = """class MyClass:
    def __init__(self):
        self.value = 0

    def calculate(self, x):
        '''
        Calculates a value based on x.
        '''
        if x > 10:
            return x * self.value
        else:
            return x + self.value"""
    reduced_code, reduction_map = reduce_code(code)
    assert reduced_code == expected_reduced_code
    # Verify map entries for specific lines
    assert reduction_map[3]["type"] == "removed_comment" # # Initialize something
    assert reduction_map[4]["reduced_content"] == "        self.value = 0" # # Default value removed
    assert reduction_map[8]["reduced_content"] == "        if x > 10:" # # Condition removed
    assert reduction_map[11]["reduced_content"] == "            return x + self.value" # # Addition operation removed
    assert reduction_map[12]["type"] == "removed_comment" # # End of method